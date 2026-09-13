import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
DATA_DIR = "./jaundice_dataset"
OUT_DIR = "outputs"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)

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

print("Loading dataset...")
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
print(f"Found {len(full_dataset)} images, classes: {full_dataset.classes}")

label_map = {i: name for i, name in enumerate(full_dataset.classes)}
with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

indices = list(range(len(full_dataset)))
random.shuffle(indices)

n = len(indices)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
train_idx, val_idx, test_idx = indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(datasets.ImageFolder(DATA_DIR, transform=eval_transform), val_idx)
test_dataset = Subset(datasets.ImageFolder(DATA_DIR, transform=eval_transform), test_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

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

    print(f"Epoch {epoch}/{EPOCHS} | train_loss {train_loss_avg:.4f} | val_loss {val_loss_avg:.4f} | val_acc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "model_jaundice.pth"))
        print(f"  -> saved best model (val_acc {val_acc:.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="train_loss")
ax1.plot(history["val_loss"], label="val_loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.legend()
ax1.set_title("Loss")
ax2.plot(history["val_acc"], label="val_acc")
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.legend()
ax2.set_title("Validation Accuracy")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)

model.load_state_dict(torch.load(os.path.join(OUT_DIR, "model_jaundice.pth"), weights_only=True))
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
report = classification_report(all_labels, all_preds, target_names=full_dataset.classes)
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(full_dataset.classes)
ax.set_yticklabels(full_dataset.classes)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
fig.colorbar(im)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)

with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Test Accuracy: {acc:.4f}\n\n")
    f.write(report)
    f.write(f"\nConfusion Matrix:\n{cm}\n")

print("\n=== TEST RESULTS ===")
print(f"Test Accuracy: {acc:.4f}")
print(report)
print(f"Confusion Matrix:\n{cm}")
print(f"\nArtifacts saved in '{OUT_DIR}':")
print("  - model_jaundice.pth")
print("  - label_map.json")
print("  - training_curves.png")
print("  - confusion_matrix.png")
print("  - classification_report.txt")