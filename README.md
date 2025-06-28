# Generative models for car anonymization in street scene images

This repository is the code base of the paper titled "Balancing privacy and utility: An evaluation of generative models for car anonymization in street scene images".
It allows you to compare various image generation models for anonymizing cars. The following models are used: **DP-GAN, OASIS, Kolors**, and **Stable Diffusion XL (SDXL)**.

---

## Table of Contents
1. [Setup](#setup)
   1. [Kolors](#kolors)
   2. [DP-GAN](#dp_gan)
   3. [OASIS](#oasis)
   4. [SDXL](#sdxl)
2. [Preparing Cityscapes](#preparing-cityscapes)
3. [Generating Masks](#generating-masks)
4. [Generating Images](#generating-images)
   1. [Kolors – Steps & Whole](#kolors)
   2. [SDXL – Steps & Whole](#sdxl-1)
   3. [DP-GAN](#dp_gan-1)
   4. [OASIS](#oasis-1)

---

## Setup<a name="setup"></a>

### Kolors<a name="kolors"></a>
Follow the official instructions to clone the **Kolors-Inpainting** repository and set up the appropriate Conda environment:
<https://github.com/Kwai-Kolors/Kolors/blob/master/inpainting/README.md>

Alternatively, you can load the model directly from Hugging Face: <https://huggingface.co/Kwai-Kolors/Kolors-Inpainting>

---

### DP-GAN<a name="dp_gan"></a>
1. Clone the repository: <https://github.com/sj-li/DP_GAN>
2. Download the checkpoints (ZIP) and unpack them into **`./checkpoints/`**. The structure should look like this:

```bash
DP_GAN/
├── checkpoints/
│ └── dp_gan_cityscapes/
└── scripts/
```

---

### OASIS<a name="oasis"></a>
* Repository & Cityscapes Checkpoint: <https://github.com/boschresearch/OASIS>

---

### SDXL<a name="sdxl"></a>
```bash
conda create -n sdxl python=3.10
conda activate sdxl
pip install -r requirementsSDXL.txt
```

---

## Preparing Cityscapes<a name="preparing-cityscapes"></a>
To convert and split the Cityscapes dataset, use the instructions from SPADE:
<https://github.com/NVlabs/SPADE>

---

## Generating Masks<a name="generating-masks"></a>
With **`mask_generator.py`**, you can create binary masks (class *car*) and copy the associated RGB images.

```bash
conda activate sdxl
python mask_generator.py \
 --input_dir /datasets/cityscapes/val \
 --output_dir ./outputs/val_masks
```

This will create the following folders:

| Folder | Description |
| --- | --- |
| `outputs/val_masks/` | Main output directory |
| `├── images/<city>/*_leftImg8bit.png` | RGB images |
| `└── masks/<city>/*_gtFine_binary.png` | Binary masks |

---

## Generating Images<a name="generating-images"></a>

### Kolors<a name="kolors"></a>
* **Steps** (Inpainting each car individually)

```bash
conda activate Kolors
python kolors_steps.py \
 --input_dir ./outputs/val_masks \
 --output_dir ./outputs/kolors_steps
```

* **Whole** (Complete mask in one step)

```bash
conda activate Kolors
python kolors_whole.py \
 --input_dir ./outputs/val_masks \
 --output_dir ./outputs/kolors_whole
```

---

### SDXL<a name="sdxl-1"></a>
* **Steps**
```bash
conda activate sdxl
python sdxl_steps.py \
 --input_dir ./outputs/val_masks \
 --output_dir ./outputs/sdxl_steps
```

* **Whole**
```bash
conda activate sdxl
python sdxl_whole.py \
 --input_dir ./outputs/val_masks \
 --output_dir ./outputs/sdxl_whole
```

---

### DP-GAN<a name="dp_gan-1"></a>
1. Follow the original documentation to generate images: <https://github.com/sj-li/DP_GAN>
2. Merge with original images & masks:

```bash
python combineGANpics.py \
 --ganpics /path/to/dpgan_results \
 --originalpics ./outputs/val_masks/images \
 --masks ./outputs/val_masks/masks \
 --output ./outputs/dpgan_merged
```

---

### OASIS<a name="oasis-1"></a>
1. Follow the original documentation to generate images: <https://github.com/boschresearch/OASIS>
2. Merge step analogous to DP-GAN:

```bash
python combineGANpics.py \
 --ganpics /path/to/oasis_results \
 --originalpics ./outputs/val_masks/images \
 --masks ./outputs/val_masks/masks \
 --output ./outputs/oasis_merged
```