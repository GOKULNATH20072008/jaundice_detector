import argparse
import json
import os
import sys

import torch
from PIL import Image
from torchvision import models, transforms

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


def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()
    return model


def predict(model, image):
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        prob = torch.softmax(logits, dim=1)[0]
        pred_class = int(logits.argmax(dim=1).item())
        confidence = float(prob[pred_class].item())
    return pred_class, confidence, prob.tolist()


def main():
    parser = argparse.ArgumentParser(description="Classify a jaundice/healthy eye image.")
    parser.add_argument("image", help="Path to an eye image")
    parser.add_argument(
        "--model",
        default=os.path.join("outputs", "model_jaundice.pth"),
        help="Path to saved model weights",
    )
    parser.add_argument(
        "--label-map",
        default=os.path.join("outputs", "label_map.json"),
        help="Path to label map JSON",
    )
    args = parser.parse_args()

    model = load_model(args.model)

    with open(args.label_map, "r") as f:
        label_map = json.load(f)

    image = Image.open(args.image).convert("RGB")
    tensor = transform(image)

    pred_class, confidence, probs = predict(model, tensor)
    class_name = label_map[str(pred_class)]
    result = "JAUNDICE DETECTED" if class_name == "jaundice_eye" else "HEALTHY"

    print(f"Image:      {args.image}")
    print(f"Class:      {class_name}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Result:     {result}")


if __name__ == "__main__":
    main()