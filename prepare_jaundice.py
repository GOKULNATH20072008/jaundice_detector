import os
from datasets import load_dataset
from PIL import Image

# 1. Define folder paths
base_dir = "./jaundice_dataset"
jaundice_dir = os.path.join(base_dir, "jaundice_eye")
healthy_dir = os.path.join(base_dir, "healthy_eye")

os.makedirs(jaundice_dir, exist_ok=True)
os.makedirs(healthy_dir, exist_ok=True)

print("Downloading dataset from Hugging Face...")
# 2. Load dataset
dataset = load_dataset("Bleachcarte/Jaundice_Dataset", split="train")

print("Saving images into sorted folders...")
# 3. Save images to respective directories
for idx, item in enumerate(dataset):
    image = item["image"]
    label = item["label"]  # 1 = Jaundice, 0 = Healthy (or vice-versa depending on dataset mapping)

    # Check image mode and convert if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    if label == 1:
        save_path = os.path.join(jaundice_dir, f"jaundice_{idx}.jpg")
    else:
        save_path = os.path.join(healthy_dir, f"healthy_{idx}.jpg")

    image.save(save_path)

print(f"Done! Files saved in '{base_dir}' directory.")