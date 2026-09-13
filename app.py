import io
import json
import os

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "model_jaundice.pth")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "outputs", "label_map.json")
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}

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

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location="cpu"))
model.eval()

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)


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

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.softmax(logits, dim=1)[0]
        pred_class = int(logits.argmax(dim=1).item())
        confidence = float(prob[pred_class].item())

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