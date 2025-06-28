#!/usr/bin/env python3
"""Kolors Inpainting"""

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
from __future__ import annotations

from datetime import datetime
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.utils import load_image
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import (
    StableDiffusionXLInpaintPipeline,
)
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer

# -------------------------------------------------------------
# Argument‑Parsing
# -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True, type=str, help="Output folder for generated images")
parser.add_argument("--input_dir", required=True, type=str, help="Cityscapes folder (images/ + masks/)")
args = parser.parse_args()

image_output_dir = args.output_dir
base_dir = args.input_dir

cities = ["frankfurt", "lindau", "munster"]

# -------------------------------------------------------------
# Runtime
# -------------------------------------------------------------
now = datetime.now()
print(f"date and time: {now:%Y-%m-%d %H:%M:%S}")
print(f"GPU COUNT {torch.cuda.device_count()}")
print(f"Current Python Environment: {sys.prefix}")

# -------------------------------------------------------------
# collect images
# -------------------------------------------------------------
pictures: dict[str, list[str]] = {c: [] for c in cities}
for city in cities:
    city_dir = os.path.join(base_dir, "images", city)
    for fn in os.listdir(city_dir):
        if fn.endswith("_leftImg8bit.png"):
            pictures[city].append(fn.replace("_leftImg8bit.png", ""))
print(f"{sum(len(v) for v in pictures.values())} images loaded")

# -------------------------------------------------------------
# Kolors weights loaded
# -------------------------------------------------------------
ckpt_dir = "./Kolors/weights/Kolors-Inpainting"
text_encoder = ChatGLMModel.from_pretrained(f"{ckpt_dir}/text_encoder", torch_dtype=torch.float16).half()
tokenizer = ChatGLMTokenizer.from_pretrained(f"{ckpt_dir}/text_encoder")
vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae").half()
scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet").half()

pipe = StableDiffusionXLInpaintPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
).to("cuda")

# -------------------------------------------------------------
# Prompts and Parameters
# -------------------------------------------------------------
prompt = (
    "A photorealistic car seamlessly integrated into an urban street scene, with consistent texture and lighting. "
    "Replace the original car with a modern, neutral design, ensuring no logos, license plates, or text. The new car "
    "should have a generic color, blending naturally with realistic reflections and shadows."
)
negative_prompt = (
    "worst quality, low resolution, overexposed, blurry, distorted shapes, numbers on doors, text artifacts, "
    "unrealistic shadows."
)

generator = torch.Generator(device="cpu").manual_seed(42)

# -------------------------------------------------------------
# create output directory
# -------------------------------------------------------------
os.makedirs(image_output_dir, exist_ok=True)
for city in cities:
    os.makedirs(os.path.join(image_output_dir, city), exist_ok=True)

# -------------------------------------------------------------
# main loop
# -------------------------------------------------------------
zeit_total = 0.0
processed = 0

for city in cities:
    for base in pictures[city]:
        json_path = os.path.join(base_dir, "masks", city, f"{base}_gtFine_polygons.json")
        img_path = os.path.join(base_dir, "images", city, f"{base}_leftImg8bit.png")

        in_image = load_image(img_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # put together the car mask
        mask = np.zeros((1024, 2048), dtype=np.uint8)
        for obj in data["objects"]:
            if obj["label"] == "car":
                poly = np.array(obj["polygon"], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [poly], 255)
        in_mask = Image.fromarray(mask)

        start = time.time()
        result = pipe(
            prompt=prompt,
            image=in_image,
            mask_image=in_mask,
            height=1024,
            width=2048,
            guidance_scale=6.0,
            generator=generator,
            num_inference_steps=25,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            strength=0.999,
        ).images[0]
        zeit_total += time.time() - start

        # combine result with original
        gt_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        res_np = np.asarray(result)
        combined = cv2.add(
            cv2.bitwise_and(gt_rgb, gt_rgb, mask=cv2.bitwise_not(mask)),
            cv2.bitwise_and(res_np, res_np, mask=mask),
        )

        out_path = os.path.join(image_output_dir, city, f"{base}.png")
        Image.fromarray(combined).save(out_path)

        processed += 1
        print(f"Image saved to: {out_path}")
        print(f"time passed: {zeit_total:.2f} s for {processed} images")

print("End")
