import cv2
import numpy as np
import json
import os
from PIL import Image
import argparse

# argument parsing
parser = argparse.ArgumentParser(description="create binary masks from cityscapes labels")
parser.add_argument('--input_dir', type=str, required=True, help='Path to Cityscapes-Val-directory')
parser.add_argument('--output_dir', type=str, required=True, help='output directory for images and masks')
args = parser.parse_args()

base_path = args.input_dir
base_dir = args.input_dir

# read city sub-folders
cities = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# create output directories
image_output_dir = os.path.join(args.output_dir, "images")
mask_output_dir = os.path.join(args.output_dir, "masks")

os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)


#base_path = "/work/rn583pgoa-workdir/Cityscapes/val/"
#base_dir="/work/rn583pgoa-workdir/Cityscapes/val/"
#cities = ['frankfurt', 'lindau', 'munster']

#image_output_dir = "/work/rn583pgoa-workdir/val/images"
#mask_output_dir = "/work/rn583pgoa-workdir/val/masks"

#os.makedirs(image_output_dir, exist_ok=True)
#os.makedirs(mask_output_dir, exist_ok=True)

# array for file names
pictures = {}

for city in cities:
    city_dir = os.path.join(base_path, city)
    
    # create sub-arrays
    pictures[city] = []
    for filename in os.listdir(city_dir):
        if filename.endswith("_gtFine_polygons.json"):
            base_name = filename.split("_gtFine_polygons.json")[0]
            pictures[city].append(base_name)

#print(pictures)

# create binary masks for each images
for city in cities:
    for base_name in pictures[city]:
        # load image and the corresponding json file
        json_path = os.path.join(base_dir, city, f"{base_name}_gtFine_polygons.json")
        image_path = os.path.join(base_dir, city, 'leftImg8bit', f"{base_name}_leftImg8bit.png")           
        # load original
        image_og = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_og, cv2.COLOR_BGR2RGB)

        # create empty mask
        mask = np.zeros((1024, 2048), dtype=np.uint8)
        
        # load polygon
        with open(json_path, 'r') as f:
            results = json.load(f)

        all_polygons = []
        
        # iterate over all objects and take polygons
        for item in results["objects"]:
            if item["label"] == "car":
                polygon_points = np.array(item['polygon'], np.int32)
                polygon_points = polygon_points.reshape((-1, 1, 2))
                all_polygons.append(polygon_points)

        # add all polygons to the mask
        for polygon in all_polygons:
            cv2.fillPoly(mask, [polygon], 255)

        # save mask as png
        mask_save_path = os.path.join(mask_output_dir, f"{base_name}_gtFine_binary.png")
        mask_image = Image.fromarray(mask)
        mask_image.save(mask_save_path)

        # save the original to the mask
        image_save_path = os.path.join(image_output_dir, f"{base_name}_leftImg8bit.png")
        cv2.imwrite(image_save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        print(f"mask saved: {mask_save_path}")
        print(f"original saved: {image_save_path}")
