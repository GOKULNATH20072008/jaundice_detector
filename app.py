import io
import json
import logging
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

from eye_gate import gate_is_real_eye  # noqa: E402

logger = logging.getLogger(__name__)

torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "model_jaundice.pth")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "outputs", "label_map.json")
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}
LOW_CONFIDENCE = 0.92
MIN_IMG_EDGE = 100
MAX_IMG_EDGE = 2600
MIN_EYE_BOX_RATIO = 0.05
MIN_EYE_BOX_PX = 20
EYE_PADDING = 0.35

# image-quality thresholds (calibrated on the dataset eye regions)
BLUR_LAPLACIAN_VAR = 50.0
MIN_REGION_MEAN = 55.0
MAX_REGION_MEAN = 210.0
MIN_REGION_STD = 18.0

MSG_NON_EYE = "No valid human eye detected. Please upload a clear photo of a human eye."
MSG_QUALITY = "Image quality is insufficient. Please upload a clearer, well-lit eye photo."
MSG_MODEL_ERROR = "Unable to analyze this image. Please try again."

EYE_GATE_ENABLED = os.environ.get("EYE_GATE_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

eye_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
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


def get_eye_box(image):
    gray = np.asarray(image.convert("L"))
    best = None
    h, w = gray.shape
    targets = [(gray, 1), (cv2.resize(gray, (max(32, w * 2), max(32, h * 2))), 2)]
    for target, scale in targets:
        for cascade in get_eye_cascades():
            for x, y, bw, bh in cascade.detectMultiScale(target, scaleFactor=1.1, minNeighbors=1, minSize=(12, 12)):
                area = (bw / scale) * (bh / scale)
                if best is None or area > best[4]:
                    best = (int(x / scale), int(y / scale), int(bw / scale), int(bh / scale), area)
    return best


def is_eye_image(image):
    return get_eye_box(image) is not None


def extract_eye_region(image):
    box = get_eye_box(image)
    if box is None:
        return None
    x, y, w, h, _ = box
    pad = int(EYE_PADDING * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.width, x + w + pad)
    y1 = min(image.height, y + h + pad)
    return image.crop((x0, y0, x1, y1))


def assess_quality(region, eye_box):
    _, _, ew, eh, _ = eye_box
    if min(ew, eh) < MIN_EYE_BOX_PX:
        return "eye_too_small"

    gray = np.asarray(region.convert("L"))
    grayf = gray.astype(np.float32)

    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_LAPLACIAN_VAR:
        return "blurry"
    mean_bright = grayf.mean()
    if mean_bright < MIN_REGION_MEAN:
        return "too_dark"
    if mean_bright > MAX_REGION_MEAN:
        return "too_bright"
    if grayf.std() < MIN_REGION_STD:
        return "low_contrast"
    return "ok"


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

    width, height = image.size
    if min(width, height) < MIN_IMG_EDGE or max(width, height) > MAX_IMG_EDGE:
        return jsonify(
            {
                "error": (
                    f"Unsupported image size {width}x{height}px. "
                    f"Please upload an eye photo between {MIN_IMG_EDGE} and "
                    f"{MAX_IMG_EDGE} pixels on each side."
                )
            }
        ), 400

    if not EYE_GATE_ENABLED:
        logger.warning("EYE_GATE_ENABLED=false — skipping semantic eye gate, falling back to Haar-cascade-only behavior")
    else:
        gate_result = gate_is_real_eye(image)
        if not gate_result["is_real_human_eye"]:
            logger.info("Gate rejected image: %s", gate_result.get("reason"))
            return jsonify(
                {
                    "result": "NOT_AN_EYE",
                    "tone": "warning",
                    "warning": MSG_NON_EYE,
                }
            )

    eye_box = get_eye_box(image)
    if eye_box is None:
        return jsonify(
            {
                "result": "NOT_AN_EYE",
                "tone": "warning",
                "warning": MSG_NON_EYE,
            }
        )

    x, y, w, h, _ = eye_box
    eye_size_ratio = min(w, h) / min(width, height)
    if eye_size_ratio < MIN_EYE_BOX_RATIO:
        return jsonify(
            {
                "result": "EYE_TOO_SMALL",
                "tone": "warning",
                "warning": (
                    "The eye is too small in the photo — it looks zoomed out. "
                    "Please upload a close-up of the eye."
                ),
            }
        )

    region = extract_eye_region(image)
    quality = assess_quality(region, eye_box)
    if quality != "ok":
        return jsonify({"result": "POOR_QUALITY", "tone": "warning", "warning": MSG_QUALITY})

    try:
        model = get_model()
        label_map = get_label_map()
        tensor = eye_transform(region).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.softmax(logits, dim=1)[0]
            confidence = float(prob[1].item())
    except Exception:
        return jsonify({"result": "MODEL_ERROR", "tone": "error", "warning": MSG_MODEL_ERROR})

    if confidence < LOW_CONFIDENCE and float(prob[0].item()) < LOW_CONFIDENCE:
        return jsonify(
            {
                "result": "UNCERTAIN",
                "tone": "warning",
                "warning": "The photo is not clear enough for a reliable result. Please upload a clear, well-lit close-up of the eye.",
                "confidence": round(max(confidence, float(prob[0].item())), 4),
                "probabilities": {
                    label_map["0"]: round(float(prob[0]), 4),
                    label_map["1"]: round(float(prob[1]), 4),
                },
            }
        )

    if confidence >= LOW_CONFIDENCE:
        result = "JAUNDICE DETECTED"
        pred_class = 1
        confidence = confidence
    else:
        result = "HEALTHY"
        pred_class = 0
        confidence = float(prob[0].item())

    class_name = label_map[str(pred_class)]

    return jsonify(
        {
            "class": class_name,
            "result": result,
            "tone": "ok",
            "confidence": round(confidence, 4),
            "probabilities": {
                label_map["0"]: round(float(prob[0]), 4),
                label_map["1"]: round(float(prob[1]), 4),
            },
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)