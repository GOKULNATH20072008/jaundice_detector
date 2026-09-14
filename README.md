# Jaundice Detector — AI Eye Screening

A prototype web application that screens for jaundice from a **close-up human
eye photo**. Built with Flask + PyTorch (ResNet-18) + OpenCV. **No API keys
required** — everything runs locally with open-source libraries and local model
weights.

This is a **screening aid / prototype, not a diagnostic medical device**.
It is not a substitute for professional medical evaluation.

---

## Architecture

```
        UPLOADED IMAGE
              ↓
    1. IMAGE VALIDATION   (file type · size bounds)
              ↓
    2. HUMAN EYE DETECTION (OpenCV Haar cascades → box or "no eye")
              ↓
    3. EYE REGION EXTRACTION (crop eye + padding)
              ↓
    4. IMAGE QUALITY CHECK  (blur · brightness · contrast · size)
              ↓
    5. JAUNDICE CLASSIFICATION (ResNet-18 → healthy / jaundice)
              ↓
        SCREENING RESULT
```

The pipeline is implemented in `pipeline.py` as clearly separated stages:

| Stage | Function | Responsibility |
|-------|----------|----------------|
| 1 | `validate_image_dims()` | Image size / bounds check (rejects tiny or huge uploads) |
| 2 | `detect_eye(image)` | "Is there a valid human eye?" → returns box or `None` |
| 3 | `crop_eye(image, box)` | Extract the eye region with padding |
| 4 | `check_image_quality(crop, box)` | "Is the crop good enough to analyze?" |
| 5 | `classify_eye(crop)` | "Given a valid eye image, does it resemble jaundice?" |
| — | `analyze_image(image)` | Runs the full pipeline, returns a structured result |

Nothing on the user-facing page is a fake "99%" output; the pipeline either
returns a `status` of `invalid`, `poor_quality`, `uncertain`, or `success`.

---

## Dataset

- **Source:** [Bleachcarte/Jaundice_Dataset](https://huggingface.co/datasets/Bleachcarte/Jaundice_Dataset)
- **Size:** 256 images (141 healthy eyes, 115 jaundiced eyes).
- **Type:** close-up human eye photographs (with sclera, iris, eyelid visible).
- **Loading:** `prepare_jaundice.py` downloads the HF dataset and sorts images
  into `jaundice_dataset/healthy_eye/` and `jaundice_dataset/jaundice_eye/`.

**Limitation:** the dataset is very small (256 images). It is enough to
demonstrate the pipeline but **not** enough to draw robust clinical conclusions.
Reported metrics are honest measurements on a held-out split, not claims of
medical accuracy.

---

## Model

- **Architecture:** ResNet-18, ImageNet-pretrained backbone, replaced `fc` head
  with a 2-class linear layer.
- **Weight file:** `outputs/model_jaundice.pth` (committed to the repo).
- **Labels:** `outputs/label_map.json` → `{"0": "healthy_eye", "1": "jaundice_eye"}`.

### Reported metrics (held-out test set, 39 images)

```
Test Accuracy:  0.8974
                precision  recall  f1-score
 healthy_eye        0.90    0.90     0.90
jaundice_eye        0.89    0.89     0.89
```

Confusion matrix:

```
pred→        healthy  jaundice
healthy      19       2
jaundice      2      16
```

> ⚠️ These are measurements on a small held-out split. The full-dataset pass
> rate for classification is ~84%; the remaining images are correctly routed to
> `invalid` (no eye) / `poor_quality` / `uncertain`.

### Training

```bash
pip install -r requirements.txt        # runtime deps
pip install datasets scikit-learn      # training-only deps

python prepare_jaundice.py             # download + sort dataset
python scripts/train_jaundice.py       # retrain (stratified 70/15/15 split)
```

The training script:
1. Loads the dataset and prints the class distribution (141 / 115).
2. Performs a **stratified** train/validation/test split (70 / 15 / 15).
3. Uses transfer learning (ImageNet weights) with a 2-class head.
4. Trains for 20 epochs, saves the best model by validation accuracy.
5. Evaluates on the *held-out* test split and writes `precision`, `recall`,
   `F1`, and a confusion matrix to `outputs/classification_report.txt`.

---

## Preprocessing

Identical transforms between training and inference:

```
RGB conversion → Resize(256) → CenterCrop(224) → ToTensor → Normalize
(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # ImageNet stats
```

- `eval_transform` in `pipeline.py` matches the training eval transform exactly.
- The CNN receives **only the cropped eye region** — never an arbitrary full
  image containing a face/background/object.

---

## Eye detection

- **Method:** OpenCV Haar cascades (`haarcascade_eye.xml`, glasses, left/right
  eye variants), run at 1× and 2× scale, largest-area detection wins.
- **Why:** no external service, no API key, lightweight, works on CPU.
- **Face photos:** eyes must occupy ≥ 10% of the image (`MIN_EYE_BOX_RATIO`);
  a full-face or distant photo is rejected as `invalid`.
- Anything else — posters, artwork, animals, objects, food, landscapes — is
  rejected at this stage with **no jaundice prediction**.

---

## Quality gate

Before classification, the eye crop is checked for:

- minimum detection size (`MIN_EYE_BOX_PX`)
- excessive blur (`Laplacian` variance < threshold)
- extreme darkness / brightness (mean pixel bounds)
- low contrast (std-dev floor)

If any check fails, the classifier is **not** run and the response is
`poor_quality`:

> "Image quality is insufficient. Please upload a clearer, well-lit eye photo."

---

## Inference flow

`analyze_image(image)` returns a structured dict:

```json
{
  "status": "success",
  "eye_detected": true,
  "image_quality": "acceptable",
  "prediction": "jaundice_detected",
  "model_confidence": 0.96,
  "message": "Model confidence: 96.4%. Jaundice eye.",
  "disclaimer": "AI screening prototype — not a medical diagnosis."
}
```

For invalid images:

```json
{
  "status": "invalid",
  "eye_detected": false,
  "prediction": null,
  "message": "No human eye detected. Please upload a clear photograph of a human eye.",
  "disclaimer": "AI screening prototype — not a medical diagnosis."
}
```

For poor-quality images:

```json
{
  "status": "poor_quality",
  "eye_detected": true,
  "prediction": null,
  "message": "Image quality is insufficient. Please upload a clearer, well-lit eye photo.",
  "disclaimer": "AI screening prototype — not a medical diagnosis."
}
```

---

## API

- `GET /` — HTML UI.
- `POST /predict` — multipart form with `image` field. Validated MIME type
  (`png/jpeg/webp/bmp`), max upload 8 MB, corrupt files rejected.

---

## Confidence

The raw softmax output is **not** presented as a clinical probability.
The UI shows "Model confidence: 96.5%" and the note "Model output — not a
clinical probability". No calibration has been performed; the thresholds were
chosen empirically from the held-out split.

---

## Railway deployment

- **Platform:** Railway (gunicorn via `Procfile`, single worker).
- **Runtime:** `gunicorn app:app --workers 1 --threads 2 --timeout 120 --max-requests 500 --max-requests-jitter 50`
- **CPU-only** inference with `torch.set_num_threads(1)` and OMP/MKL threads
  pinned to 1 to stay within memory limits.
- **Zero environment variables required.** `model_jaundice.pth` is committed to
  the repo, so no download step at build time.

---

## Privacy

- Uploaded images are held **in memory only**; no persistent storage of patient
  images. Files are read into `BytesIO` and discarded after inference.
- No data is sent to any external API or third-party service.

---

## Test cases (automated suite)

`tests/test_pipeline.py` runs `analyze_image()` directly (no server, no API):
`python tests/test_pipeline.py`

| # | Case | Expected |
|---|------|----------|
| 1 | Clear healthy eye | `success`, `prediction: healthy` |
| 2 | Clear jaundice eye | `success`, `prediction: jaundice_detected` |
| 3 | Blurry eye | `poor_quality` (no prediction) |
| 4 | Very dark eye | `poor_quality` / `invalid` (no prediction) |
| 5 | Very bright eye | `poor_quality` / `invalid` (no prediction) |
| 6 | Face photo | `invalid` (no prediction) |
| 7 | Ganesha / painted image | `invalid` (no prediction) |
| 8 | Random object | `invalid` (no prediction) |
| 9 | Food image | `invalid` (no prediction) |
| 10 | Landscape / distant | `invalid` (no prediction) |
| 11 | Poster | `invalid` (no prediction) |
| 12 | Emoji / meme | `invalid` (no prediction) |

Known limitation: a photograph of a screen displaying a real eye photo can still
reach classification (the eye is genuine, just mediated); the crop/quality/
confidence gates are the defense for the remaining edge cases.

---

## Disclaimer

This is a **prototype screening aid**, not a diagnostic medical device.
Do not use it in place of a doctor's examination, laboratory testing, or
professional judgement.