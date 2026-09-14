"""Reproducible training script for the jaundice eye classifier.

Architecture : ResNet-18 (ImageNet pre-trained → 2-class head)
Dataset      : Bleachcarte/Jaundice_Dataset (256 images: 141 healthy, 115 jaundice)
Split        : Stratified 70/15/15 (train/val/test)

Outputs written to ``outputs/``:
  - model_jaundice.pth          best weights (by validation accuracy)
  - label_map.json              class index → name mapping
  - training_curves.png         loss / accuracy curves
  - confusion_matrix.png        held-out test confusion matrix
  - classification_report.txt   precision / recall / F1

Note: with only 256 images the results are a screening prototype, NOT a
clinically validated diagnostic tool.  Explicitly state this in documentation.
"""

import json
import os
import random
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "jaundice_dataset")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)

# ── transforms ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ── dataset ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
full_dataset = datasets.ImageFolder(DATA_DIR, transform=None)
class_names = full_dataset.classes
print(f"Found {len(full_dataset)} images, classes: {class_names}")

labels = [s[1] for s in full_dataset.samples]
print("Class distribution:", dict(Counter(labels)))

# save label map
label_map = {i: name for i, name in enumerate(class_names)}
with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

# stratified split: 70 / 15 / 15
from collections import defaultdict
by_class = defaultdict(list)
for idx, lab in enumerate(labels):
    by_class[lab].append(idx)

train_idx, val_idx, test_idx = [], [], []
for c, idxs in by_class.items():
    random.shuffle(idxs)
    n = len(idxs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx.extend(idxs[:n_train])
    val_idx.extend(idxs[n_train : n_train + n_val])
    test_idx.extend(idxs[n_train + n_val :])

random.shuffle(train_idx)
random.shuffle(val_idx)
random.shuffle(test_idx)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# wrap with transforms
full_dataset.transform = train_transform
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(
    datasets.ImageFolder(DATA_DIR, transform=eval_transform), val_idx
)
test_dataset = Subset(
    datasets.ImageFolder(DATA_DIR, transform=eval_transform), test_idx
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# ── model ───────────────────────────────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# ── training loop ───────────────────────────────────────────────────────────
best_val_acc = 0.0
history = {"train_loss": [], "val_loss": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    train_loss_avg = running_loss / len(train_idx)
    val_loss_avg = val_loss / len(val_idx)
    val_acc = val_correct / val_total

    history["train_loss"].append(train_loss_avg)
    history["val_loss"].append(val_loss_avg)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train_loss {train_loss_avg:.4f} | "
        f"val_loss {val_loss_avg:.4f} | "
        f"val_acc {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "model_jaundice.pth"))
        print(f"  -> saved best model (val_acc {val_acc:.4f})")

# ── plots ───────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="train_loss")
ax1.plot(history["val_loss"], label="val_loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.set_title("Loss")
ax2.plot(history["val_acc"], label="val_acc")
ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.legend(); ax2.set_title("Val Acc")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)

# ── test evaluation ─────────────────────────────────────────────────────────
model.load_state_dict(
    torch.load(os.path.join(OUT_DIR, "model_jaundice.pth"), weights_only=True)
)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=class_names)

# confusion matrix plot
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig.colorbar(im); fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)

# classification report file
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Test Accuracy:  {acc:.4f}\n")
    f.write(f"Test Precision: {prec:.4f}\n")
    f.write(f"Test Recall:    {rec:.4f}\n")
    f.write(f"Test F1:        {f1:.4f}\n\n")
    f.write(report)
    f.write(f"\nConfusion Matrix:\n{cm}\n")

print("\n=== TEST RESULTS ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1:        {f1:.4f}")
print(report)
print(f"Confusion Matrix:\n{cm}")
print(f"\nArtifacts written to '{OUT_DIR}/'")
