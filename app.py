import io
import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402
from PIL import Image  # noqa: E402

from pipeline import analyze_image  # noqa: E402

torch.set_num_threads(1)
logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if file.content_type not in ALLOWED_TYPES:
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image"}), 400

    result = analyze_image(image)
    logger.info(
        "predict status=%s prediction=%s confidence=%s",
        result.get("status"),
        result.get("prediction"),
        result.get("model_confidence"),
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
