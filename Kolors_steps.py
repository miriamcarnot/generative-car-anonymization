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
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from diffusers.utils import load_image 
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import (
    StableDiffusionXLInpaintPipeline,
)
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer

# -------------------------------------------------------------
# color‑escape‑sequences & Start‑Log
# -------------------------------------------------------------
RED = "\033[91m"
WHITE = "\033[0m"

now = datetime.now()
print(f"Date and Time: {RED}{now:%Y-%m-%d %H:%M:%S}{WHITE}")

print(f"GPU COUNT {RED}{torch.cuda.device_count()}{WHITE}")
print(f"Current Python Environment: {RED}{sys.prefix}{WHITE}")

# -------------------------------------------------------------
# Argument‑Parsing
# -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str)
parser.add_argument("--input_dir", type=str)
args = parser.parse_args()

image_output_dir = args.output_dir
base_dir = args.input_dir

cities = ["frankfurt", "lindau", "munster"]

print(f"image_output_dir: {RED}'{image_output_dir}'{WHITE}.")

# -------------------------------------------------------------
# get file names
# -------------------------------------------------------------
pictures: dict[str, list[str]] = {}
counter_imgs = 0

for city in cities:
    city_dir = os.path.join(base_dir, "images", city)
    pictures[city] = []
    for filename in os.listdir(city_dir):
        counter_imgs += 1
        if filename.endswith("_leftImg8bit.png"):
            pictures[city].append(filename.replace("_leftImg8bit.png", ""))

print(f"{counter_imgs} images loaded")

# -------------------------------------------------------------
# Pipeline – Modelle laden
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
# Prompts & Parameters
# -------------------------------------------------------------
prompt = (
    "A photorealistic car seamlessly integrated into an urban street scene, with consistent texture and lighting. "
    "Replace the original car with a modern, neutral design, ensuring no logos, license plates, or text. The new car "
    "has to fill the binary segmentation mask completely."
)
negative_prompt = (
    "worst quality, low resolution, overexposed, blurry, distorted shapes, numbers on doors, text artifacts, "
    "unrealistic shadows."
)

generator = torch.Generator(device="cpu").manual_seed(42)

# -------------------------------------------------------------
# create output dir
# -------------------------------------------------------------
if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)
    for city in cities:
        os.makedirs(os.path.join(image_output_dir, city))
    print(f"Folder {RED}'{image_output_dir}'{WHITE} and sub-folders created.")
else:
    print(f"Folder {RED}'{image_output_dir}'{WHITE} already exists.")

# -------------------------------------------------------------
# main loop: car inpainting
# -------------------------------------------------------------
zeit_total = 0.0
car_counter_total = 0

for city in cities:
    for base_name in pictures[city]:
        json_path = os.path.join(base_dir, "masks", city, f"{base_name}_gtFine_polygons.json")
        image_path = os.path.join(base_dir, "images", city, f"{base_name}_leftImg8bit.png")

        in_image = load_image(image_path)

        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        polygons = [
            np.array(obj["polygon"], np.int32).reshape((-1, 1, 2))
            for obj in results["objects"]
            if obj["label"] == "car"
        ]

        gt_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = np.zeros((1024, 2048), dtype=np.uint8)

        img_before = in_image
        img_car_count = 0
        img_time = 0.0

        for poly in polygons:
            cv2.fillPoly(mask, [poly], 255)
            in_mask = Image.fromarray(mask)

            start = time.time()
            result = pipe(
                prompt=prompt,
                image=img_before,
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
            img_time += time.time() - start

            result_np = np.array(result)
            img_car_count += 1
            Image.fromarray(result_np).save(
                os.path.join(image_output_dir, city, f"{base_name}_{img_car_count}.png")
            )

            # Verschmelzen mit Original
            masked = cv2.bitwise_and(result_np, result_np, mask=mask)
            inv_mask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(gt_rgb, gt_rgb, mask=inv_mask)
            gt_rgb = cv2.add(background, masked)

            img_before = Image.fromarray(gt_rgb)
            mask.fill(0)
            car_counter_total += 1

        zeit_total += img_time
        Image.fromarray(gt_rgb).save(os.path.join(image_output_dir, city, f"{base_name}.png"))

        print(f"Image saved to: {os.path.join(image_output_dir, city, base_name + '.png')}")
        print(f"Time passed: {zeit_total:.2f} seconds for {car_counter_total} cars")

print("End")
