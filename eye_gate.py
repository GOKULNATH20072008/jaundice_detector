import base64
import io
import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
GATE_TIMEOUT = 10
MODEL = "claude-sonnet-4-6"


def _load_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def gate_available():
    """True only if a real API key is configured. When False the caller should
    fall back to the existing Haar-cascade pipeline instead of blocking eyes."""
    return bool(_load_api_key())

SYSTEM_PROMPT = """\
You are a strict image gatekeeper for a medical screening tool. You will be shown one image.

Answer ONLY with a JSON object, no other text:
{"is_real_human_eye": true|false, "reason": "<one short phrase>"}

Rules for true:
- The image must be a real photograph (not a drawing, painting, print, poster, or screenshot)
- It must show a close-up of an actual human eye, with visible sclera (white part), iris, and eyelid
- The eye must belong to a real living human, photographed directly (not a photo of a photo, not a magazine/poster image of an eye, not a doll/mannequin/statue eye)

Answer false for anything else, including: full-face or full-body photos where the eye is not the clear subject, animal eyes, cartoon/anime/CGI eyes, sunglasses or closed eyes, printed/painted/drawn eyes, screenshots, text, unrelated objects, or blank/corrupted images.

Be strict — when uncertain, answer false."""


def _image_to_base64(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _parse_gate_response(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end])
    except (json.JSONDecodeError, ValueError):
        return None


def gate_is_real_eye(image):
    api_key = _load_api_key()
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — gate disabled, failing closed")
        return {"is_real_human_eye": False, "reason": "gate_api_error"}

    b64 = _image_to_base64(image)

    payload = {
        "model": MODEL,
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": "Is this a real human eye photo? Respond with JSON only."},
                ],
            }
        ],
        "system": SYSTEM_PROMPT,
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        resp = requests.post(
            ANTHROPIC_API_URL, json=payload, headers=headers, timeout=GATE_TIMEOUT
        )
        resp.raise_for_status()
        body = resp.json()
        text = body["content"][0]["text"]
        parsed = _parse_gate_response(text)
        if parsed is None:
            logger.warning("Gate returned unparseable response: %s", text[:200])
            return {"is_real_human_eye": False, "reason": "gate_parse_error"}
        return {
            "is_real_human_eye": bool(parsed.get("is_real_human_eye", False)),
            "reason": str(parsed.get("reason", "unknown")),
        }
    except requests.Timeout:
        logger.warning("Gate API timed out")
        return {"is_real_human_eye": False, "reason": "gate_api_error"}
    except Exception as exc:
        logger.warning("Gate API error: %s", exc)
        return {"is_real_human_eye": False, "reason": "gate_api_error"}
