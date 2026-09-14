import io
import json
import os
import threading

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import models, transforms  # noqa: E402

torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "model_jaundice.pth")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "outputs", "label_map.json")
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}
LOW_CONFIDENCE = 0.55

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

app = Flask(__name__)

_model = None
_label_map = None
_eye_cascades = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location="cpu"))
            model.eval()
            _model = model
    return _model


def get_label_map():
    global _label_map
    with _model_lock:
        if _label_map is None:
            with open(LABEL_MAP_PATH, "r") as f:
                _label_map = json.load(f)
    return _label_map


def get_eye_cascades():
    global _eye_cascades
    with _model_lock:
        if _eye_cascades is None:
            cascade_dir = os.path.join(os.path.dirname(cv2.__file__), "data")
            _eye_cascades = [
                cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_eye.xml")),
                cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_eye_tree_eyeglasses.xml")),
                cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_lefteye_2splits.xml")),
                cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_righteye_2splits.xml")),
            ]
    return _eye_cascades


def is_eye_image(image):
    gray = np.asarray(image.convert("L"))
    for cascade in get_eye_cascades():
        eyes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(12, 12))
        if len(eyes) > 0:
            return True
    return False


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

    if not is_eye_image(image):
        return jsonify(
            {
                "result": "NOT_AN_EYE",
                "warning": "No eye detected in the image. Please upload a clear photo of the eye.",
            }
        )

    model = get_model()
    label_map = get_label_map()

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.softmax(logits, dim=1)[0]
        pred_class = int(logits.argmax(dim=1).item())
        confidence = float(prob[pred_class].item())

    if confidence < LOW_CONFIDENCE:
        return jsonify(
            {
                "result": "UNCERTAIN",
                "warning": "The image is unclear. Please upload a clear, well-lit photo of the eye.",
                "confidence": round(confidence, 4),
                "probabilities": {
                    label_map["0"]: round(float(prob[0]), 4),
                    label_map["1"]: round(float(prob[1]), 4),
                },
            }
        )

    class_name = label_map[str(pred_class)]
    result = "JAUNDICE DETECTED" if class_name == "jaundice_eye" else "HEALTHY"

    return jsonify(
        {
            "class": class_name,
            "result": result,
            "confidence": round(confidence, 4),
            "probabilities": {
                label_map["0"]: round(float(prob[0]), 4),
                label_map["1"]: round(float(prob[1]), 4),
            },
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)