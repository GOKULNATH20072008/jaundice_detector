"""Pipeline integration tests (Phase 12 test cases).

All tests exercise ``analyze_image()`` directly — no Flask test client needed,
no live API calls, no external services.  Run with:
    python tests/test_pipeline.py
"""

import os
import sys
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import analyze_image, detect_eye  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jaundice_dataset"
)


def load_fixture(name):
    return Image.open(os.path.join(FIXTURES, name)).convert("RGB")


def load_dataset(name, subdir="healthy_eye"):
    return Image.open(os.path.join(DATASET, subdir, name)).convert("RGB")


def make_dark(image, factor=0.2):
    arr = np.array(image).astype(np.float32) * factor
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def make_bright(image, factor=2.2):
    arr = np.array(image).astype(np.float32) * factor
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def make_food():
    img = Image.new("RGB", (400, 400), (240, 235, 230))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.ellipse([50, 50, 350, 350], outline="gray", width=4)
    for cx, cy, r, c in [(150, 150, 40, (200, 50, 30)),
                          (250, 180, 50, (50, 160, 50)),
                          (180, 270, 45, (220, 180, 30))]:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
    return img


class TestHealthyEye(unittest.TestCase):
    """Case 1: clear healthy eye → success, prediction=healthy."""

    def test_healthy_eye_detected_and_healthy(self):
        im = load_dataset("healthy_50.jpg", "healthy_eye")
        r = analyze_image(im)
        self.assertEqual(r["status"], "success")
        self.assertTrue(r["eye_detected"])
        self.assertEqual(r["prediction"], "healthy")
        self.assertGreater(r["model_confidence"], 0.80)


class TestJaundiceEye(unittest.TestCase):
    """Case 2: clear jaundice eye → success, prediction=jaundice_detected."""

    def test_jaundice_eye_detected(self):
        im = load_dataset("jaundice_142.jpg", "jaundice_eye")
        r = analyze_image(im)
        self.assertEqual(r["status"], "success")
        self.assertTrue(r["eye_detected"])
        self.assertEqual(r["prediction"], "jaundice_detected")
        self.assertGreater(r["model_confidence"], 0.80)


class TestBlurryEye(unittest.TestCase):
    """Case 3: blurry eye → poor_quality."""

    def test_blurry_eye_rejected(self):
        im = load_fixture("eye_imperfect.jpg")
        r = analyze_image(im)
        self.assertIn(r["status"], ("poor_quality", "invalid"))
        self.assertNotEqual(r["status"], "success")


class TestVeryDarkEye(unittest.TestCase):
    """Case 4: very dark eye → poor_quality or invalid."""

    def test_dark_eye_rejected(self):
        base = load_fixture("eye_light.jpg")
        dark = make_dark(base, factor=0.15)
        r = analyze_image(dark)
        self.assertIn(r["status"], ("poor_quality", "invalid"))
        self.assertNotEqual(r["status"], "success")


class TestVeryBrightEye(unittest.TestCase):
    """Case 5: very bright/overexposed eye → poor_quality or invalid."""

    def test_bright_eye_rejected(self):
        base = load_fixture("eye_light.jpg")
        bright = make_bright(base, factor=2.5)
        r = analyze_image(bright)
        self.assertIn(r["status"], ("poor_quality", "invalid"))
        self.assertNotEqual(r["status"], "success")


class TestFaceImage(unittest.TestCase):
    """Case 6: face photo (eye visible but not the clear subject) → invalid."""

    def test_face_photo_rejected(self):
        im = load_fixture("full_face.jpg")
        r = analyze_image(im)
        self.assertEqual(r["status"], "invalid")
        self.assertFalse(r["eye_detected"])


class TestGaneshaImage(unittest.TestCase):
    """Case 7: Ganesha / painted image → invalid, NO prediction."""

    def test_ganesha_rejected(self):
        im = load_fixture("painting.jpg")
        r = analyze_image(im)
        self.assertEqual(r["status"], "invalid")
        self.assertFalse(r["eye_detected"])
        self.assertIsNone(r["prediction"])


class TestRandomObject(unittest.TestCase):
    """Case 8: random object (mug) → invalid."""

    def test_object_rejected(self):
        im = load_fixture("object_mug.jpg")
        r = analyze_image(im)
        self.assertEqual(r["status"], "invalid")
        self.assertFalse(r["eye_detected"])
        self.assertIsNone(r["prediction"])


class TestFoodImage(unittest.TestCase):
    """Case 9: food image → invalid."""

    def test_food_rejected(self):
        im = make_food()
        r = analyze_image(im)
        self.assertEqual(r["status"], "invalid")
        self.assertFalse(r["eye_detected"])
        self.assertIsNone(r["prediction"])


class TestLandscape(unittest.TestCase):
    """Case 10: landscape / distant photo → invalid."""

    def test_landscape_rejected(self):
        im = load_fixture("full_body.jpg")
        r = analyze_image(im)
        self.assertEqual(r["status"], "invalid")
        self.assertFalse(r["eye_detected"])
        self.assertIsNone(r["prediction"])


class TestPoster(unittest.TestCase):
    """Poster of an eye → invalid."""

    def test_poster_rejected(self):
        im = load_fixture("poster.jpg")
        r = analyze_image(im)
        self.assertEqual(r["status"], "invalid")
        self.assertFalse(r["eye_detected"])


class TestEmoji(unittest.TestCase):
    """Eye emoji / meme → invalid."""

    def test_emoji_rejected(self):
        im = load_fixture("emoji.jpg")
        r = analyze_image(im)
        self.assertEqual(r["status"], "invalid")
        self.assertFalse(r["eye_detected"])


class TestPhotoOfScreen(unittest.TestCase):
    """Photo of a screen showing an eye → status is success OR invalid (documented
    limitation).  Verify the pipeline does not crash."""

    def test_photo_of_screen_handled(self):
        im = load_fixture("photo_of_screen.jpg")
        r = analyze_image(im)
        self.assertIn(r["status"], ("success", "invalid", "uncertain", "poor_quality"))
        self.assertIsNotNone(r["message"])
        self.assertIn("disclaimer", r)


class TestStructuredResultShape(unittest.TestCase):
    """Every result dict has the required keys."""

    def test_all_keys_present(self):
        required = {
            "status", "eye_detected", "image_quality", "prediction",
            "model_confidence", "message", "disclaimer",
        }
        for name in ["eye_light.jpg", "full_face.jpg", "painting.jpg"]:
            im = load_fixture(name)
            r = analyze_image(im)
            self.assertTrue(required.issubset(set(r.keys())), f"missing keys for {name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
