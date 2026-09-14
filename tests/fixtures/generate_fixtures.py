"""Generate tests/fixtures/ sample images for the eye-gate test suite.

Real eye cases are derived from the training dataset (jaundice_dataset/);
non-eye cases are synthesized so the fixtures are self-contained.
Run from the repo root:  python tests/fixtures/generate_fixtures.py
"""

import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(os.path.dirname(ROOT), "jaundice_dataset")
SIZE = (400, 400)

os.makedirs(FIXTURES, exist_ok=True)


def copy_case(name, src_path, transform=None):
    im = Image.open(src_path).convert("RGB")
    if transform:
        im = transform(im)
    im = im.resize(SIZE)
    out = os.path.join(FIXTURES, name)
    im.save(out, quality=90)
    print("wrote", name)


def pick_healthy(index):
    return os.path.join(DATASET, "healthy_eye", f"healthy_{index}.jpg")


def pick_jaundice(index):
    return os.path.join(DATASET, "jaundice_eye", f"jaundice_{index}.jpg")


# --- 1-5: genuine close-up eye photos (single or slightly-processed) ---
copy_case("eye_light.jpg", pick_healthy(50))
copy_case("eye_dark.jpg", pick_healthy(133))  # bucketed separately via warm/dark grade
g = lambda im: im.point(lambda p: p * 0.82)  # noqa: E731  darker-skin grade
copy_case("eye_dark.jpg", pick_healthy(133), g)
copy_case("eye_jaundice.jpg", pick_jaundice(142))
copy_case("eye_imperfect.jpg", pick_healthy(98), lambda im: im.filter(ImageFilter.GaussianBlur(2)))
copy_case("eye_angle.jpg", pick_healthy(140), lambda im: im.rotate(12, expand=False, fillcolor="black"))


def new_img(bg=(245, 245, 245)):
    return Image.new("RGB", SIZE, bg)


def draw(im):
    return ImageDraw.Draw(im)


# --- 6: poster / print of an eye ---
im = new_img((240, 235, 228))
d = draw(im)
d.rounded_rectangle([30, 30, 370, 250], radius=12, fill=(210, 205, 198), outline=(80, 80, 80), width=3)
d.ellipse([120, 80, 270, 210], fill=(255, 250, 240))                 # sclera
d.ellipse([160, 110, 235, 185], fill=(110, 70, 40))                  # iris
d.ellipse([185, 128, 215, 158], fill=(20, 20, 20))                   # pupil
d.rectangle([120, 80, 272, 96], fill=(200, 195, 188))                # lid line
for i in range(8):                                                   # printed eyelash strokes
    x = 150 + i * 18
    d.line([(x, 96), (x - 12, 46)], fill=(40, 40, 40), width=3)
d.rectangle([100, 270, 300, 310], fill=(70, 70, 70))
d.text((118, 278), "OPTIC GALLERY", fill=(240, 240, 240))
d.rectangle([100, 320, 300, 360], fill=(180, 60, 50))
d.text((150, 330), "AUCTION", fill=(245, 245, 245))
im.save(os.path.join(FIXTURES, "poster.jpg"), quality=90)

# --- 7: magazine / stock-crop of an eye with caption ---
im = new_img((220, 216, 210))
d = draw(im)
d.rectangle([10, 10, 390, 300], fill=(245, 240, 232))
d.ellipse([120, 70, 285, 220], fill=(255, 250, 240))
d.ellipse([165, 105, 245, 185], fill=(80, 120, 170))
d.ellipse([195, 130, 220, 155], fill=(15, 15, 15))
d.rectangle([118, 70, 287, 86], fill=(235, 228, 220))
d.text((50, 320), "Eyes of the silver screen", fill=(30, 30, 30))
d.text((60, 355), "Full feature, p. 42", fill=(120, 120, 120))
im.save(os.path.join(FIXTURES, "magazine.jpg"), quality=90)

# --- 8: painted / drawn eye (illustration) ---
im = new_img((252, 246, 236))
d = draw(im)
for cx, cy, r, c in [(200, 180, 70, (92, 142, 120)), (175, 165, 55, (120, 168, 150)),
                     (200, 200, 40, (60, 110, 92)), (185, 175, 25, (160, 200, 170))]:
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
d.polygon([(130, 120), (270, 120), (300, 140), (100, 140)], fill=(70, 90, 80))
d.arc([120, 60, 290, 240], start=0, end=180, fill=(40, 40, 40), width=4)  # brush lid stroke
im.save(os.path.join(FIXTURES, "painting.jpg"), quality=90)


def draw_person_facing(draw, y_top, y_bot, x_center=SIZE[0] // 2, skin=(205, 170, 130), scale=1.0):
    face_h = y_bot - y_top
    half_w = int(face_h * 0.42 * scale)
    draw.polygon([(x_center - half_w, y_bot), (x_center - int(half_w * 0.9), y_top), (x_center + int(half_w * 0.9), y_top), (x_center + half_w, y_bot)], fill=skin)
    return half_w


# --- 9: full-face photo (eye visible but not the subject) ---
im = new_img((185, 165, 145))
d = draw(im)
hw = draw_person_facing(d, 40, 360)
ec = 250
d.ellipse([ec - 12, 150, ec + 12, 170], fill=(255, 250, 245))  # tiny eye
d.ellipse([ec - 6, 155, ec + 6, 166], fill=(60, 45, 30))
d.rectangle([ec - 60, 340, ec + 60, 400], fill=(30, 40, 80))    # shoulders
d.rectangle([ec - 120, 190, ec + 120, 200], fill=(90, 75, 55))  # brow
im.save(os.path.join(FIXTURES, "full_face.jpg"), quality=90)

# --- 10: full-body / distant photo ---
im = new_img((170, 190, 215))
d = draw(im)
d.rectangle([0, 0, 400, 200], fill=(150, 190, 220))   # sky
d.rectangle([0, 200, 400, 400], fill=(90, 140, 90))   # grass
d.ellipse([120, 40, 250, 170], fill=(255, 200, 130))  # body block (no visible face detail)
d.polygon([(120, 80), (250, 80), (250, 200), (120, 200)], fill=(70, 60, 120))  # torso
im.save(os.path.join(FIXTURES, "full_body.jpg"), quality=90)

# --- 11: ear ---
im = new_img()
d = draw(im)
d.polygon([(120, 90), (190, 80), (240, 150), (200, 260), (90, 240), (80, 160)], fill=(205, 170, 135))
d.polygon([(120, 130), (150, 120), (165, 190), (110, 210)], fill=(170, 138, 105))
d.arc([80, 90, 230, 240], start=90, end=270, fill=(150, 120, 90), width=4)
im.save(os.path.join(FIXTURES, "ear.jpg"), quality=90)

# --- 12: animal (cat) eye ---
im = new_img((120, 150, 90))
d = draw(im)
d.ellipse([50, 40, 160, 150], outline=(30, 30, 30), width=6)      # lid
d.ellipse([70, 55, 130, 120], fill=(235, 170, 60))                # golden iris
d.polygon([(100, 60), (97, 95), (103, 95)], fill=None)            # vertical slit
im.save(os.path.join(FIXTURES, "animal_eye.jpg"), quality=90)

# --- 13: cartoon / anime eye ---
im = new_img((255, 240, 250))
d = draw(im)
d.ellipse([40, 40, 360, 220], fill=(255, 255, 255))               # oversized lid
d.ellipse([90, 70, 310, 250], fill=(140, 210, 255))               # huge iris
d.ellipse([170, 110, 235, 175], fill=(40, 40, 40))                # pupil
d.ellipse([185, 118, 205, 140], fill=(255, 255, 255))             # star highlight
d.ellipse([255, 190, 270, 205], fill=(255, 255, 255))             # sparkle
for i in range(10):
    x = 60 + i * 32
    d.line([(x, 60), (x - 8, 0)], fill=(30, 30, 30), width=6)     # lashes
d.ellipse([90, 200, 310, 245], fill=(244, 200, 210))              # blush
im.save(os.path.join(FIXTURES, "cartoon.jpg"), quality=90)

# --- 14: doll / mannequin eye ---
im = new_img((220, 214, 208))
d = draw(im)
d.ellipse([55, 60, 345, 230], fill=(235, 228, 220))
d.ellipse([120, 100, 280, 220], fill=(150, 190, 230))             # vitreous doll iris
d.ellipse([175, 140, 225, 190], fill=(35, 35, 45))                # piped pupil
d.ellipse([190, 150, 205, 165], fill=(255, 255, 255))
d.line([(55, 60), (345, 60)], fill=(60, 60, 60), width=3)         # molded lid seam
im.save(os.path.join(FIXTURES, "doll.jpg"), quality=90)

# --- 15: sunglasses ---
im = new_img()
d = draw(im)
for cx in (115, 285):
    d.polygon([(cx - 45, 100), (cx + 45, 100), (cx + 55, 240), (cx - 55, 240)], fill=(25, 25, 30))
    d.ellipse([cx - 55, 95, cx + 55, 245], outline=(25, 25, 30), width=6)
d.line([(115, 150), (285, 150)], fill=(90, 90, 95), width=8)      # bridge
d.line([(160, 200), (290, 310)], fill=(90, 90, 95), width=6)      # arm
im.save(os.path.join(FIXTURES, "sunglasses.jpg"), quality=90)

# --- 16: closed eye ---
im = new_img((205, 170, 135))
d = draw(im)
d.arc([60, 120, 340, 160], start=180, end=360, fill=(60, 45, 35), width=8)  # closed lid crease
for i in range(12):
    x = 80 + i * 22
    d.line([(x, 140), (x - 6, 60)], fill=(25, 25, 25), width=2)   # lashes
im.save(os.path.join(FIXTURES, "closed_eye.jpg"), quality=90)

# --- 17: heavily blurred/unrecognizable ---
im = Image.open(pick_healthy(66)).convert("RGB").filter(ImageFilter.GaussianBlur(28))
im.resize(SIZE).save(os.path.join(FIXTURES, "heavy_blur.jpg"), quality=90)

# --- 18: screenshot of text / UI ---
im = new_img((28, 32, 48))
d = draw(im)
for i in range(8):
    d.rectangle([20, 40 + i * 40, 380, 60 + i * 40], fill=(60 + i * 8, 70, 100 + i * 8))
d.rectangle([20, 360, 180, 388], fill=(60, 140, 90))
im.save(os.path.join(FIXTURES, "screenshot.jpg"), quality=90)

# --- 19: random object (mug) ---
im = new_img((210, 200, 190))
d = draw(im)
d.rectangle([110, 120, 290, 340], fill=(120, 90, 160), outline=(80, 60, 110), width=4)
d.ellipse([110, 100, 290, 150], fill=(135, 105, 175))
d.arc([290, 130, 380, 250], start=-90, end=90, fill=(100, 80, 140), width=10)  # handle
d.ellipse([160, 180, 250, 260], fill=(245, 225, 150))                        # emblem
im.save(os.path.join(FIXTURES, "object_mug.jpg"), quality=90)

# --- 20: solid color / blank ---
Image.new("RGB", SIZE, (128, 128, 128)).save(os.path.join(FIXTURES, "blank.jpg"), quality=90)

# --- 21: photo of a screen displaying an eye photo ---
photo = Image.open(pick_jaundice(200)).convert("RGB").resize((280, 280))
frame = Image.new("RGB", (400, 400), (45, 45, 52))
d = draw(frame)
d.rectangle([20, 20, 380, 340], fill=(220, 220, 225), outline=(70, 70, 70), width=4)  # screen bezel
frame.paste(photo, (66, 60))
d.rectangle([20, 340, 380, 370], fill=(60, 60, 70))                                  # stand
d.text((150, 346), "image.jpg", fill=(220, 220, 220))
frame.save(os.path.join(FIXTURES, "photo_of_screen.jpg"), quality=90)

# --- 22: emoji / meme style image ---
im = new_img((255, 230, 170))
d = draw(im)
d.ellipse([60, 40, 340, 320], outline=(200, 160, 80), width=12)   # big yellow face
d.ellipse([130, 120, 175, 165], fill=(40, 40, 40))                # emoji eyes
d.ellipse([225, 120, 270, 165], fill=(40, 40, 40))
d.arc([150, 200, 250, 260], start=10, end=170, fill=(60, 45, 30), width=8)  # mouth
im.save(os.path.join(FIXTURES, "emoji.jpg"), quality=90)

print("done")