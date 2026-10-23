import io
import os
import gc
import requests
import numpy as np
from PIL import Image, ImageFilter
import onnxruntime as ort
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# সুপারফাস্ট ৪.৫ MB মডেল (১.৫ সেকেন্ডে প্রসেস করবে)
MODEL_PATH = "u2netp.onnx"
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    print("Downloading fast U2Netp model...")
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(MODEL_URL, headers=headers, stream=True, timeout=60)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    print("Download complete!")

# আল্ট্রা-ফাস্ট লাইটওয়েট সেটিংস
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.enable_cpu_mem_arena = False
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

@app.route("/keep-alive", methods=["GET"])
def keep_alive():
    return "OK", 200

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "running"}), 200

@app.route("/removebg", methods=["POST"])
def remove_background():
    img_bytes = None
    if request.files:
        file = request.files.get("image") or next(iter(request.files.values()))
        img_bytes = file.read()
    elif request.data:
        img_bytes = request.data

    if not img_bytes:
        return jsonify({"error": "No image"}), 400

    try:
        original_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        orig_w, orig_h = original_img.size

        # দ্রুত প্রসেসিংয়ের জন্য সাইজ নিয়ন্ত্রণ (ম্যাক্সিমাম ৮০০ পিক্সেল)
        if orig_w > 800 or orig_h > 800:
            original_img.thumbnail((800, 800), Image.Resampling.BILINEAR)
            orig_w, orig_h = original_img.size

        rgb_img = original_img.convert("RGB")
        resized = rgb_img.resize((320, 320), Image.Resampling.BILINEAR)

        # ইমেজনেট স্ট্যান্ডার্ড নরমালাইজেশন
        arr = np.array(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (arr - mean) / std

        tensor = np.transpose(norm, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        # ইনফারেন্স রান (সুপারফাস্ট)
        ort_outs = session.run(None, {input_name: tensor})
        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)
        pred = (pred - mi) / (ma - mi + 1e-8)
        pred = np.squeeze(pred)

        # কড়া ব্যাকগ্রাউন্ড কাটআউট (সাদা ব্যাকগ্রাউন্ড সম্পূর্ণ মুছে ফেলা)
        pred_clean = np.where(pred < 0.38, 0.0, pred)
        alpha = np.clip((pred_clean - 0.38) / (0.78 - 0.38), 0.0, 1.0) * 255.0
        
        mask_pil = Image.fromarray(alpha.astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)

        # ফটোশপ চোক (সাদা বর্ডার ৩ পিক্সেল ভেতর দিকে কেটে বাদ দেওয়া)
        mask_eroded = mask_pil.filter(ImageFilter.MinFilter(3))
        mask_final = mask_eroded.filter(ImageFilter.GaussianBlur(radius=0.7))

        original_img.putalpha(mask_final)

        # ফাস্ট কম্প্রেশন
        output_io = io.BytesIO()
        original_img.save(output_io, format="PNG", compress_level=1)
        output_io.seek(0)

        del tensor, ort_outs, pred, original_img, rgb_img, mask_pil, mask_eroded, mask_final
        gc.collect()

        return send_file(output_io, mimetype="image/png")

    except Exception as e:
        gc.collect()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
