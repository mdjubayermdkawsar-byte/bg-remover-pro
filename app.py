import io
import os
import gc
import urllib.request
import numpy as np
from PIL import Image, ImageFilter
import onnxruntime as ort
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# সুপারফাস্ট ৪.৫ MB মডেল (১.৫ সেকেন্ডে রান হবে)
MODEL_PATH = "u2netp.onnx"
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    print("Downloading fast U2Netp model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete!")

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

        # দ্রুত প্রসেসিংয়ের জন্য অপ্টিমাইজড সাইজ
        if orig_w > 800 or orig_h > 800:
            original_img.thumbnail((800, 800), Image.Resampling.BILINEAR)
            orig_w, orig_h = original_img.size

        rgb_img = original_img.convert("RGB")
        resized = rgb_img.resize((320, 320), Image.Resampling.BILINEAR)

        # ইমেজনেট নরমালাইজেশন
        arr = np.array(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (arr - mean) / std

        tensor = np.transpose(norm, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        # এআই ইনফারেন্স (মাত্র ১.৫ সেকেন্ড)
        ort_outs = session.run(None, {input_name: tensor})
        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)
        pred = (pred - mi) / (ma - mi + 1e-8)
        pred = np.squeeze(pred)

        # জিরো-হোয়াইট থ্রেশহোল্ড (সাদা ব্যাকগ্রাউন্ড সম্পূর্ণ ডিলিট)
        pred_clean = np.where(pred < 0.38, 0.0, pred)
        alpha = np.clip((pred_clean - 0.38) / (0.78 - 0.38), 0.0, 1.0) * 255.0
        
        mask_pil = Image.fromarray(alpha.astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)

        # ফটোশপ চোক (সাদা বর্ডার ৩ পিক্সেল ভেতর দিকে কেটে ফেলা)
        mask_eroded = mask_pil.filter(ImageFilter.MinFilter(3))
        mask_final = mask_eroded.filter(ImageFilter.GaussianBlur(radius=0.7))

        original_img.putalpha(mask_final)

        # ফাস্ট পিএনজি সেভ
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
