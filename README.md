# Real-ESRGAN Fine-Tuning Guide

This guide provides step-by-step instructions for fine-tuning the Real-ESRGAN model for image super-resolution, optimized for the USR-248 underwater image dataset but transferable to other domains (e.g., terrestrial, medical images). Real-ESRGAN enhances low-resolution images to high-resolution with improved visual quality. This README covers setup, dataset preparation, fine-tuning, inference, and troubleshooting, designed for Jupyter Notebook or Google Colab workflows with GPU suppo

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup Instructions](#setup-instructions)
4. [Optional Steps](#optional-steps)
5. [Fine-Tuning Configuration](#fine-tuning-configuration)
6. [Training the Model](#training-the-model)
7. [Running Inference](#running-inference)
8. [Verifying Outputs](#verifying-outputs)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

## Overview

Real-ESRGAN (Real Enhanced Super-Resolution Generative Adversarial Network) is an advanced model for image super-resolution, capable of upscaling low-resolution images by a factor of 4 while preserving details and reducing artifacts. This guide focuses on fine-tuning the pre-trained Real-ESRGAN model on a custom dataset (USR-248) using a Python 3.7 environment, leveraging Conda for dependency management and a GPU (e.g., T4 on Colab) for faster computation.

The fine-tuning process uses 5,000 iterations instead of the recommended 40,000 due to the limited dataset size and computational resources. Since the model is pre-trained on similar data, 5,000 iterations are sufficient for effective performance.

## Prerequisites

* **Hardware**: A machine with a GPU (e.g., NVIDIA T4 on Google Colab) is recommended for faster training and inference.
* **Software**:

  * Python 3.7
  * Conda (via `condacolab` for Colab environments)
  * Git
  * Jupyter Notebook or Colab
* **Dataset**: Paired dataset in `datasets/hr` and `datasets/lr`
* **Repository**: The Real-ESRGAN repo is included as a Git submodule within this project, no need to clone it separately.

## Setup Instructions

1. **Install CondaColab (for Colab)**:

   ```python
   !pip install -q condacolab
   import condacolab
   condacolab.install()
   ```

2. **Verify CondaColab**:

   ```python
   import condacolab
   condacolab.check()
   ```

3. **Create Conda Environment**:

   ```bash
   !conda create -n realesrgan37 python=3.7 -y
   ```

4. **Verify Python Version**:

   ```bash
   !conda run -n realesrgan37 python --version
   ```

5. **Navigate to Submodule Directory**:

   ```bash
   %cd realesrgan/
   ```

6. **Install Dependencies**:

   ```bash
   !conda run -n realesrgan37 pip install -r requirements.txt
   ```

7. **Install Package in Development Mode**:

   ```bash
   !conda run -n realesrgan37 python setup.py develop
   ```

## Optional Steps

1. **Download Pre-Trained Models**:

   ```bash
   !wget -c https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
   !wget -c https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
   !wget -c https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth -P experiments/pretrained_models
   ```

2. **Generate Meta Info File**:

   ```bash
   !conda run -n realesrgan37 python scripts/generate_meta_info_pairdata.py \
     --input datasets/hr datasets/lr \
     --meta_info datasets/meta_info_usr248_pair.txt
   ```

## Fine-Tuning Configuration

Use the `options/finetune_realesrgan_x4plus.yml` config. This config sets `total_iter` to 5000 and loads from pre-trained weights. Dataset used: `USR-248`.

Ensure this file is copied or edited as needed from the original config file. For full YAML content, refer to the detailed documentation in the repository.

## Training the Model

```bash
!conda run -n realesrgan37 python realesrgan/train.py \
  -opt options/finetune_realesrgan_x4plus.yml \
  --auto_resume
```

* Model checkpoints saved in: `experiments/finetune_RealESRGANx4plus_5000iters/models`

## Running Inference

```bash
!conda run -n realesrgan37 python inference_realesrgan.py -n RealESRGAN_x4plus \
  -i inputs \
  -o results \
  --model_path experiments/finetune_RealESRGANx4plus_5000iters/models/net_g_5000.pth \
  --outscale 4 \
  --suffix out
```

* Input: `inputs/`
* Output: `results/`, with `_out` suffix

## Verifying Outputs

```bash
!ls experiments/finetune_RealESRGANx4plus_5000iters/models
```

Ensure `net_g_5000.pth` and other model checkpoints exist.

## Troubleshooting

* Use GPU runtime on Colab.
* Missing meta info file? Regenerate using the provided script.
* Errors in training? Ensure all paths and YAML config are correct.

## References

* [Real-ESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)
* [Real-ESRGAN Paper](https://arxiv.org/abs/2107.10833)
* [CondaColab](https://github.com/conda-incubator/condacolab)

---

This document assumes the Real-ESRGAN repository is already initialized as a Git submodule within your project. Ensure all paths and links are correct if your folder structure differs.