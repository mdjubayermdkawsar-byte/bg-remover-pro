import io
import os
import gc
import requests
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

MODEL_PATH = "u2netp.onnx"
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"

# সার্ভার চালুর সময় মডেল না থাকলে ডাউনলোড করে নেবে (৪.৫ MB)
if not os.path.exists(MODEL_PATH):
    print("Downloading U2Netp model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Download complete!")

# রেন্ডারের ৫১২ MB র‍্যামের জন্য অপ্টিমাইজড
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# রেন্ডারকে জাগিয়ে রাখার জন্য
@app.route("/keep-alive", methods=["GET"])
def keep_alive():
    return "I am awake!", 200

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "running", "service": "U2Netp Render AI BG Remover"}), 200

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

        if orig_w > 1500 or orig_h > 1500:
            original_img.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
            orig_w, orig_h = original_img.size

        rgb_img = original_img.convert("RGB")
        resized = rgb_img.resize((320, 320), Image.Resampling.BILINEAR)

        arr = np.array(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (arr - mean) / std

        tensor = np.transpose(norm, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        ort_outs = session.run(None, {input_name: tensor})
        pred = ort_outs[0][:, 0, :, :]

        ma = np.max(pred)
        mi = np.min(pred)
        pred = (pred - mi) / (ma - mi + 1e-8)
        pred = np.squeeze(pred)

        alpha = np.clip((pred - 0.12) / (0.88 - 0.12), 0.0, 1.0) * 255.0
        mask_pil = Image.fromarray(alpha.astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)

        original_img.putalpha(mask_pil)

        output_io = io.BytesIO()
        original_img.save(output_io, format="PNG", optimize=True)
        output_io.seek(0)

        del tensor, ort_outs, pred, original_img, rgb_img, mask_pil
        gc.collect()

        return send_file(output_io, mimetype="image/png")

    except Exception as e:
        gc.collect()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
