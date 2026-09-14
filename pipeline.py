"""Jaundice screening pipeline.

Separate responsibilities:
  detect_eye    → find an eye in the image  (Haar cascades, local CV)
  crop_eye      → extract the eye region
  check_image_quality → is the crop suitable for analysis?
  classify_eye  → run the jaundice classifier
  analyze_image → full pipeline, returns structured result dict

No external API keys required.  Everything runs locally via OpenCV + PyTorch.
"""

import json
import os
import threading

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "model_jaundice.pth")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "outputs", "label_map.json")

# ── preprocessing must match training eval_transform exactly ────────────────
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

# ── eye detection parameters ────────────────────────────────────────────────
MIN_IMG_EDGE = 100
MAX_IMG_EDGE = 2600
MIN_EYE_BOX_RATIO = 0.10
EYE_PADDING = 0.35

# ── image-quality thresholds (calibrated on dataset eye regions) ─────────────
BLUR_LAPLACIAN_VAR = 50.0
MIN_REGION_MEAN = 55.0
MAX_REGION_MEAN = 210.0
MIN_REGION_STD = 18.0
MIN_EYE_BOX_PX = 20

# ── verdict threshold ───────────────────────────────────────────────────────
LOW_CONFIDENCE = 0.92

DISCLAIMER = "AI screening prototype — not a medical diagnosis."


# ── lazy-loaded singletons ──────────────────────────────────────────────────
_model = None
_label_map = None
_eye_cascades = None
_lock = threading.Lock()

CASCADE_NAMES = [
    "haarcascade_eye.xml",
    "haarcascade_eye_tree_eyeglasses.xml",
    "haarcascade_lefteye_2splits.xml",
    "haarcascade_righteye_2splits.xml",
]


def _get_cascades():
    global _eye_cascades
    with _lock:
        if _eye_cascades is None:
            cascade_dir = os.path.join(os.path.dirname(cv2.__file__), "data")
            _eye_cascades = [
                cv2.CascadeClassifier(os.path.join(cascade_dir, c))
                for c in CASCADE_NAMES
            ]
    return _eye_cascades


def _get_model():
    global _model
    with _lock:
        if _model is None:
            m = models.resnet18(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, 2)
            m.load_state_dict(
                torch.load(MODEL_PATH, weights_only=True, map_location="cpu")
            )
            m.eval()
            _model = m
    return _model


def _get_label_map():
    global _label_map
    with _lock:
        if _label_map is None:
            with open(LABEL_MAP_PATH) as f:
                _label_map = json.load(f)
    return _label_map


def _invalid(message):
    return {
        "status": "invalid",
        "eye_detected": False,
        "image_quality": None,
        "prediction": None,
        "model_confidence": None,
        "message": message,
        "disclaimer": DISCLAIMER,
    }


def _poor_quality(message):
    return {
        "status": "poor_quality",
        "eye_detected": True,
        "image_quality": "insufficient",
        "prediction": None,
        "model_confidence": None,
        "message": message,
        "disclaimer": DISCLAIMER,
    }


# ── stage 1: eye detection ──────────────────────────────────────────────────

def detect_eye(image):
    """Run Haar cascades at 1× and 2× scale; return the largest-area box
    found across all cascades, or *None* if no eye is detected."""
    gray = np.asarray(image.convert("L"))
    h, w = gray.shape
    targets = [(gray, 1)]
    if w * 2 <= 5000 and h * 2 <= 5000:
        targets.append((cv2.resize(gray, (w * 2, h * 2)), 2))

    best = None
    for g, scale in targets:
        for cascade in _get_cascades():
            for x, y, bw, bh in cascade.detectMultiScale(
                g, scaleFactor=1.1, minNeighbors=1, minSize=(12, 12)
            ):
                area = (bw / scale) * (bh / scale)
                if best is None or area > best[4]:
                    best = (
                        int(x / scale),
                        int(y / scale),
                        int(bw / scale),
                        int(bh / scale),
                        area,
                    )
    return best


# ── stage 2: eye region extraction ──────────────────────────────────────────

def crop_eye(image, box):
    """Crop the eye region with configurable padding."""
    x, y, w, h, _ = box
    pad = int(EYE_PADDING * max(w, h))
    return image.crop(
        (
            max(0, x - pad),
            max(0, y - pad),
            min(image.width, x + w + pad),
            min(image.height, y + h + pad),
        )
    )


# ── stage 3: image quality ──────────────────────────────────────────────────

def check_image_quality(crop, box):
    """Return {"valid": bool, "message": str}."""
    _, _, ew, eh, _ = box
    if min(ew, eh) < MIN_EYE_BOX_PX:
        return {"valid": False, "message": "The detected eye is too small."}

    gray = np.asarray(crop.convert("L"))
    grayf = gray.astype(np.float32)

    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_LAPLACIAN_VAR:
        return {"valid": False, "message": "Image is too blurry."}
    if grayf.mean() < MIN_REGION_MEAN:
        return {"valid": False, "message": "Image is too dark."}
    if grayf.mean() > MAX_REGION_MEAN:
        return {"valid": False, "message": "Image is too bright."}
    if grayf.std() < MIN_REGION_STD:
        return {"valid": False, "message": "Image has insufficient contrast."}

    return {"valid": True, "message": "Image quality acceptable."}


# ── stage 4: jaundice classifier ────────────────────────────────────────────

def classify_eye(crop):
    """Run the ResNet-18 jaundice classifier on the eye crop.
    Returns {"healthy_confidence": float, "jaundice_confidence": float}."""
    model = _get_model()
    tensor = eval_transform(crop).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.softmax(logits, dim=1)[0]
    return {
        "healthy_confidence": float(prob[0]),
        "jaundice_confidence": float(prob[1]),
    }


# ── full pipeline ───────────────────────────────────────────────────────────

def analyze_image(image):
    """Run the full screening pipeline and return a structured result dict.

    Pipeline stages:
        1. Validate image dimensions
        2. Detect eye   (Haar cascades)
        3. Crop eye region
        4. Check image quality
        5. Classify with ResNet-18 jaundice model
    """
    w, h = image.size
    if min(w, h) < MIN_IMG_EDGE or max(w, h) > MAX_IMG_EDGE:
        return _invalid(
            f"Unsupported image size {w}×{h}px. "
            f"Use an image between {MIN_IMG_EDGE} and {MAX_IMG_EDGE} pixels."
        )

    # ── stage 1: detect eye ─────────────────────────────────────────────
    box = detect_eye(image)
    if box is None:
        return _invalid(
            "No human eye detected. Please upload a clear photograph of a human eye."
        )

    x, y, bw, bh, _ = box
    eye_ratio = min(bw, bh) / min(w, h)
    if eye_ratio < MIN_EYE_BOX_RATIO:
        return _invalid(
            "The eye is too small — this looks like a face photo, not an eye "
            "close-up. Please upload a close-up of a single eye."
        )

    # ── stage 2: crop eye region ────────────────────────────────────────
    crop = crop_eye(image, box)

    # ── stage 3: quality gate ───────────────────────────────────────────
    quality = check_image_quality(crop, box)
    if not quality["valid"]:
        return _poor_quality(
            quality["message"] + " Please upload a clearer, well-lit eye photo."
        )

    # ── stage 4: jaundice classification ────────────────────────────────
    try:
        result = classify_eye(crop)
    except Exception:
        return {
            "status": "error",
            "eye_detected": True,
            "image_quality": "acceptable",
            "prediction": None,
            "model_confidence": None,
            "message": "Unable to analyze this image. Please try again.",
            "disclaimer": DISCLAIMER,
        }

    hc = result["healthy_confidence"]
    jc = result["jaundice_confidence"]
    pred_class = int(jc >= hc)
    confidence = max(hc, jc)

    if confidence < LOW_CONFIDENCE:
        return {
            "status": "uncertain",
            "eye_detected": True,
            "image_quality": "acceptable",
            "prediction": None,
            "model_confidence": round(confidence, 4),
            "message": (
                "The photo is not clear enough for a reliable result. "
                "Please upload a clear, well-lit close-up of the eye."
            ),
            "disclaimer": DISCLAIMER,
        }

    label_map = _get_label_map()
    class_name = label_map[str(pred_class)]
    prediction = "jaundice_detected" if pred_class == 1 else "healthy"
    model_label = "Jaundice" if pred_class == 1 else "Healthy"

    return {
        "status": "success",
        "eye_detected": True,
        "image_quality": "acceptable",
        "prediction": prediction,
        "model_confidence": round(confidence, 4),
        "model_label": model_label,
        "message": f"Model confidence: {confidence:.1%}. {model_label} eye.",
        "disclaimer": DISCLAIMER,
    }
