#!/usr/bin/env python3


# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import argparse
import os

import cv2
import numpy as np
from PIL import Image

# -------------------------------------------------------------
# Argument‑Parsing
# -------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Adds GAN-generated cars into original images using binary masks.",
)
parser.add_argument("--ganpics", type=str, help="Path to generated images (512×256)")
parser.add_argument("--originalpics", type=str, help="Path to original images (2048×1024)")
parser.add_argument("--masks", type=str, help="Path to binary masks (2048×1024)")
parser.add_argument("--output", type=str, help="Output folder for generated images")
args = parser.parse_args()

genPth = args.ganpics
ogPth = args.originalpics
maPth = args.masks
ouPth = args.output

print(f"Generated Images from: {genPth}")
print(f"Original Images from: {ogPth}")
print(f"Masks from: {maPth}")
print(f"Output Images to: {ouPth}")

# -------------------------------------------------------------
# Basename‑Sammlung
# -------------------------------------------------------------
base_names: list[str] = []
for filename in os.listdir(ogPth):
    if filename.endswith("_leftImg8bit.png"):
        base_names.append(filename.replace("_leftImg8bit.png", ""))
print("Images read:", len(base_names))

# -------------------------------------------------------------
# Haupt‑Loop
# -------------------------------------------------------------
for base_name in base_names:
    # Dateinamen zusammensetzen
    mask_file = f"{base_name}_gtFine_binary.png"
    img_file = f"{base_name}_leftImg8bit.png"

    # Dateien laden
    mask_np = cv2.imread(os.path.join(maPth, mask_file), cv2.IMREAD_GRAYSCALE)
    og_np_bgr = cv2.imread(os.path.join(ogPth, img_file))
    og_np = cv2.cvtColor(og_np_bgr, cv2.COLOR_BGR2RGB)

    gen_np_bgr = cv2.imread(os.path.join(genPth, img_file))  # 512×256
    gen_np = cv2.cvtColor(gen_np_bgr, cv2.COLOR_BGR2RGB)
    gen_np = cv2.resize(gen_np, (2048, 1024), interpolation=cv2.INTER_CUBIC)

    # Maske anwenden
    gen_cars = cv2.bitwise_and(gen_np, gen_np, mask=mask_np)
    inverse_mask = cv2.bitwise_not(mask_np)
    og_background = cv2.bitwise_and(og_np, og_np, mask=inverse_mask)
    final = cv2.add(og_background, gen_cars)

    # Speichern
    os.makedirs(ouPth, exist_ok=True)
    output_name = f"{base_name}_merged.png"
    Image.fromarray(final).save(os.path.join(ouPth, output_name))
    print(f"Image saved: {os.path.join(ouPth, output_name)}")

print("End")
