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

MODEL_PATH = "silueta.onnx"
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx"

# গিটহাব থেকে সুরক্ষিত স্ট্রিম ডাউনলোডার (কখনো 403 Forbidden হবে না)
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    print("Downloading high-precision Silueta model...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(MODEL_URL, headers=headers, stream=True, timeout=120)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    print("Download complete! Model size:", os.path.getsize(MODEL_PATH))

# মেমোরি সেভার ও ফাস্ট সিপিইউ সেশন (সার্ভার সারাজীবন মেমরিতে একটিভ থাকবে)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.enable_cpu_mem_arena = False
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# UptimeRobot এর পিং হ্যান্ডলার (সার্ভার কখনো ঘুমাবে না)
@app.route("/keep-alive", methods=["GET"])
def keep_alive():
    return "OK", 200

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "running", "service": "Silueta High-Precision AI BG Remover"}), 200

@app.route("/removebg", methods=["POST"])
def remove_background():
    img_bytes = None
    if request.files:
        file = request.files.get("image") or next(iter(request.files.values()))
        img_bytes = file.read()
    elif request.data:
        img_bytes = request.data

    if not img_bytes:
        return jsonify({"error": "No image data found"}), 400

    try:
        original_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        orig_w, orig_h = original_img.size

        # দ্রুত প্রসেসিংয়ের জন্য সাইজ ব্যালান্স (স্পিড ২-৩ সেকেন্ডে নামিয়ে আনবে)
        if orig_w > 900 or orig_h > 900:
            original_img.thumbnail((900, 900), Image.Resampling.BILINEAR)
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

        # এআই ইনফারেন্স
        ort_outs = session.run(None, {input_name: tensor})
        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)
        pred = (pred - mi) / (ma - mi + 1e-8)
        pred = np.squeeze(pred)

        # ১. কড়া থ্রেশহোল্ড (যাতে কোনো ব্যাকগ্রাউন্ড বা সাদা রঙ না থাকে)
        pred_clean = np.where(pred < 0.35, 0.0, pred)
        alpha = np.clip((pred_clean - 0.35) / (0.75 - 0.35), 0.0, 1.0) * 255.0
        
        mask_pil = Image.fromarray(alpha.astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)

        # ২. ফটোশপ চোক / ইরোশন: সাদা বর্ডারকে ভেতর দিকে ৩ পিক্সেল কেটে ফেলা হচ্ছে
        mask_eroded = mask_pil.filter(ImageFilter.MinFilter(3))

        # ৩. এজ মসৃণ করা (Natural Anti-aliasing)
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
