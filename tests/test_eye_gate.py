"""Unit + integration tests for the semantic "real human eye" gate.

Unit tests run WITHOUT a live API key: the Anthropic HTTP call is mocked via
``unittest.mock``. They validate that the gate always fails closed on
malformed/errored responses, that each fixture image is plumbed through to the
API, and that Flask short-circuits before any Haar cascade / model inference on
gate rejects. Run with:  python tests/test_eye_gate.py

For periodic real-API validation (requires ANTHROPIC_API_KEY) use:
  python tests/run_integration.py
"""

import io
import json
import os
import sys
import unittest
from unittest import mock

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("EYE_GATE_ENABLED", "true")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")  # noqa: S105 — mocked, never used for a real call

from PIL import Image  # noqa: E402

import eye_gate  # noqa: E402
from app import app as flask_app  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def load_fixture(name):
    return Image.open(os.path.join(FIXTURES, name)).convert("RGB")


def fake_response(text, status=200):
    body = {"content": [{"type": "text", "text": text}]}
    r = mock.Mock()
    r.json.return_value = body
    r.status_code = status
    r.raise_for_status.side_effect = None if status == 200 else requests.HTTPError(str(status))
    return r


def okay(text):
    return fake_response(text, 200)


def patched_post(text, status=200):
    return mock.patch.object(eye_gate.requests, "post", return_value=fake_response(text, status))


CASES = [
    # (fixture, expected_is_real_eye, mocked gate reason)
    ("eye_light.jpg", True, "close-up human eye, lighter skin"),
    ("eye_dark.jpg", True, "close-up human eye, darker skin"),
    ("eye_jaundice.jpg", True, "close-up human eye with jaundiced sclera"),
    ("eye_imperfect.jpg", True, "real eye photo, slightly imperfect lighting"),
    ("eye_angle.jpg", True, "real eye photo, slight off-angle"),
    ("poster.jpg", False, "printed poster of an eye"),
    ("magazine.jpg", False, "magazine stock crop of an eye"),
    ("painting.jpg", False, "painted illustration of an eye"),
    ("full_face.jpg", False, "full-face photo, eye not the subject"),
    ("full_body.jpg", False, "full-body distant photo"),
    ("ear.jpg", False, "ear, not an eye"),
    ("animal_eye.jpg", False, "animal eye"),
    ("cartoon.jpg", False, "cartoon/anime eye"),
    ("doll.jpg", False, "doll or mannequin eye"),
    ("sunglasses.jpg", False, "sunglasses cover the eye"),
    ("closed_eye.jpg", False, "closed eye"),
    ("heavy_blur.jpg", False, "heavily blurred, unrecognizable"),
    ("screenshot.jpg", False, "screenshot of UI text"),
    ("object_mug.jpg", False, "unrelated object (mug)"),
    ("blank.jpg", False, "solid color, blank"),
    ("photo_of_screen.jpg", False, "photo of a screen showing an eye photo"),
    ("emoji.jpg", False, "emoji/meme eyes"),
]


def gate_return_text(expected, reason):
    return json.dumps({"is_real_human_eye": expected, "reason": reason})


class TestGateFixturesPlumbedToAPI(unittest.TestCase):
    """Each fixture is base64-encoded, sent to the API, and answers parsed through."""

    def _run(self, fixture, expected, reason):
        with patched_post(gate_return_text(expected, reason)) as mock_post:
            result = eye_gate.gate_is_real_eye(load_fixture(fixture))
        self.assertEqual(result["is_real_human_eye"], expected, result)
        self.assertEqual(result["reason"], reason)
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "claude-sonnet-4-6")
        img_block = payload["messages"][0]["content"][0]
        self.assertEqual(img_block["source"]["media_type"], "image/jpeg")
        self.assertTrue(len(img_block["source"]["data"]) > 1000)

    def test_cases(self):
        for fixture, expected, reason in CASES:
            with self.subTest(fixture=fixture):
                self._run(fixture, expected, reason)


class TestGateParseErrorsFailClosed(unittest.TestCase):
    """A malformed / non-JSON gate answer must never let an image through."""

    def test_prose_answer(self):
        with patched_post("I can't help with that."):
            result = eye_gate.gate_is_real_eye(load_fixture("eye_light.jpg"))
        self.assertFalse(result["is_real_human_eye"])
        self.assertEqual(result["reason"], "gate_parse_error")

    def test_fenced_json_parsed(self):
        with patched_post("```json\n{\"is_real_human_eye\": false, \"reason\": \"meme\"}\n```"):
            result = eye_gate.gate_is_real_eye(load_fixture("emoji.jpg"))
        self.assertFalse(result["is_real_human_eye"])
        self.assertEqual(result["reason"], "meme")

    def test_invalid_json(self):
        with patched_post("{not json}"):
            result = eye_gate.gate_is_real_eye(load_fixture("poster.jpg"))
        self.assertFalse(result["is_real_human_eye"])
        self.assertEqual(result["reason"], "gate_parse_error")

    def test_json_missing_field_defaults_false(self):
        with patched_post(json.dumps({"reason": "no verdict field"})):
            result = eye_gate.gate_is_real_eye(load_fixture("eye_light.jpg"))
        self.assertFalse(result["is_real_human_eye"])
        self.assertEqual(result["reason"], "no verdict field")


class TestGateApiErrorsFailClosed(unittest.TestCase):
    """Timeout / HTTP / transport errors must fail closed."""

    def test_timeout(self):
        with mock.patch.object(eye_gate.requests, "post", side_effect=requests.Timeout("slow")):
            result = eye_gate.gate_is_real_eye(load_fixture("eye_light.jpg"))
        self.assertFalse(result["is_real_human_eye"])
        self.assertEqual(result["reason"], "gate_api_error")

    def test_http_500(self):
        with patched_post("server error", status=500):
            result = eye_gate.gate_is_real_eye(load_fixture("eye_light.jpg"))
        self.assertFalse(result["is_real_human_eye"])
        self.assertEqual(result["reason"], "gate_api_error")

    def test_transport_error(self):
        with mock.patch.object(eye_gate.requests, "post", side_effect=requests.ConnectionError("no")):
            result = eye_gate.gate_is_real_eye(load_fixture("eye_light.jpg"))
        self.assertFalse(result["is_real_human_eye"])
        self.assertEqual(result["reason"], "gate_api_error")

    def test_missing_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(eye_gate.requests, "post", side_effect=AssertionError("must not call API")):
                result = eye_gate.gate_is_real_eye(load_fixture("eye_light.jpg"))
        self.assertFalse(result["is_real_human_eye"])
        self.assertEqual(result["reason"], "gate_api_error")


class TestFlaskShortCircuit(unittest.TestCase):
    """Gate rejects must bypass Haar cascade + model entirely (saves compute)."""

    @classmethod
    def setUpClass(cls):
        flask_app.config["TESTING"] = True
        cls.client = flask_app.test_client()

    def _post(self, fixture):
        buf = io.BytesIO()
        load_fixture(fixture).save(buf, "JPEG")
        buf.seek(0)
        return self.client.post(
            "/predict", data={"image": (buf, fixture)}, content_type="multipart/form-data"
        )

    def test_gate_reject_shortcircuits_pipeline(self):
        with mock.patch.object(sys.modules["app"], "gate_available", return_value=True), \
             mock.patch.object(sys.modules["app"], "gate_is_real_eye", return_value={"is_real_human_eye": False, "reason": "meme"}) as gate, \
             mock.patch.object(sys.modules["app"], "get_eye_box", side_effect=AssertionError("must not run")) as haar, \
             mock.patch.object(sys.modules["app"], "get_model", side_effect=AssertionError("must not run")):
            resp = self._post("emoji.jpg")
        gate.assert_called_once()
        haar.assert_not_called()
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["result"], "NOT_AN_EYE")
        self.assertEqual(data["tone"], "warning")

    def test_no_api_key_skips_gate_and_still_finds_eye(self):
        app_mod = sys.modules["app"]
        with mock.patch.object(app_mod, "gate_available", return_value=False) as available, \
             mock.patch.object(app_mod, "gate_is_real_eye", side_effect=AssertionError("must not call gate without a key")) as gate:
            resp = self._post("eye_light.jpg")
        available.assert_called_once()
        gate.assert_not_called()
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn(data["result"], ("JAUNDICE DETECTED", "HEALTHY", "UNCERTAIN"))
        self.assertEqual(data["tone"], "ok")

    def test_gate_disabled_falls_back_to_haar(self):
        app_mod = sys.modules["app"]
        with mock.patch.object(app_mod, "gate_available", return_value=True), \
             mock.patch.object(app_mod, "gate_is_real_eye", side_effect=AssertionError("must not call gate when disabled")) as gate, \
             mock.patch.object(app_mod, "EYE_GATE_ENABLED", False):
            resp = self._post("eye_light.jpg")
        gate.assert_not_called()
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn(data["result"], ("JAUNDICE DETECTED", "HEALTHY", "UNCERTAIN"))

    def test_valid_eye_flows_to_classifier_when_gate_passes(self):
        with mock.patch.object(sys.modules["app"], "gate_available", return_value=True), \
             mock.patch.object(sys.modules["app"], "gate_is_real_eye", return_value={"is_real_human_eye": True, "reason": "close-up human eye"}):
            resp = self._post("eye_light.jpg")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn(data["result"], ("JAUNDICE DETECTED", "HEALTHY", "UNCERTAIN"))
        self.assertEqual(data["tone"], "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)