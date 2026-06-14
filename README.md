# CNN Image Super-Resolution (ESPCN)

## Project Overview

The project implements **Efficient Sub-Pixel Convolutional Neural Network (ESPCN)** for image super-resolution. Two main variants:

**Standard Model (2x)**: Grayscale upscaling, baseline for comparison.

**Experimental Model (3x)**: YCbCr color space with separate Y-channel processing, Bicubic Cb/Cr interpolation.

**Advantage**: Processing only luminance through the network reduces computation while maintaining perceptual quality, since humans are more sensitive to brightness than color changes.

---

## Technical Architecture

**Network Flow** (both models follow same pattern with different parameters):

```
Input → [Conv2d 5×5] → [ReLU/LeakyReLU] → [Residual Block 1] 
→ [Residual Block 2] → [Conv2d 3×3] → [PixelShuffle] → Output
```

**Differences**:

| Feature | Standard | Experimental |
|---------|----------|--------------|
| Upscale | 2x | 3x |
| Channels | 64×64→128×128 | 64×64→192×192 |
| Activation | ReLU | LeakyReLU(0.1) |
| Augmentation | None | Random flips + gradient clipping |
| Color | Grayscale | YCbCr (Y-channel only) |

---

## Dataset Structure

```
resources/datasets/
├── DIV2K/
│   ├── DIV2K_train_HR/           # 800 high-res training images
│   └── DIV2K_train_LR_bicubic/
│       ├── X2/                   # 2x downsampled (standard model)
│       └── X3/                   # 3x downsampled (experimental model)
├── Set5/                         # 5 benchmark images
├── Set14/                        # 14 benchmark images
└── T91/                          # 91 training images
```

**Training Split**: 800 images (0001-0800) | **Validation Split**: 100 images (0801-0900)
**Per Epoch**: 40,000 training patches | 500 validation patches

**Preprocessing**:
1. Extract Y-channel from YCbCr color space
2. Random 64×64 patches (aligned to upscale_factor)
3. Data augmentation: 50% horizontal flip + 50% vertical flip
4. Normalize to [0, 1] range

---

## Installation & Setup

**Prerequisites**: Python 3.8+, CUDA 11.0+, 8GB+ RAM, 4GB+ GPU

*conda* is recommended for running the project. Libraries can be found in requirements.txt

**Download DIV2K Dataset**: https://www.kaggle.com/datasets/anirudhmuthuswamy/div2k-hr-and-lr
Extract to `resources/datasets/DIV2K/`

---

## Usage Guide

### Training Standard Model (2x Grayscale)
```bash
python train.py
```
- **Output**: `checkpoints/best_model.pth`, `training_history.png`
- **Runtime**: depends on hardware (~20-30 minutes on 3070 + 5600X, 32GB RAM)

### Training Experimental Model (3x YCbCr)
```bash
python train_experimental.py
```
- **Output**: `checkpoints/model_experimental.pth`, `training_history_experimental.png`
- **Runtime**: depends on hardware (~20-30 minutes on 3070 + 5600X, 32GB RAM)

### Test Standard Model
```bash
python diagnose_model.py
```
- Tests on 3 validation images, compares vs. Bicubic baseline
- **Output**: `diagnose/diagnosis_image_*.png`, PSNR/SSIM metrics

### Test Experimental Model
```bash
python diagnose_model_experimental.py
```
- Same as above but for 3x model
- **Output**: `diagnose_experimental/diagnosis_image_*.png`

### Batch Upscale Images
```bash
python test_model.py
```
- Upscales all images in `img/` folder using experimental model
- **Output**: `img/upscaled/*.png` with average inference time

---

## Performance Metrics

**Expected Quality (3x Upscaling)**:
- **PSNR**: 28-31 dB (vs. HR ground truth)
- **SSIM**: 0.85-0.92
- **vs. Bicubic**: +2-4 dB PSNR improvement, +0.08-0.15 SSIM improvement

**Inference Speed**:
- **GPU**: 0.12-0.18s per image
- **CPU**: 1.2-1.8s per image
- **Model Size**: 2.1MB

**PSNR** (Peak Signal-to-Noise Ratio): Pixel-level accuracy, higher = better
**SSIM** (Structural Similarity): Perceived quality, better aligns with human perception

---

## File Structure

```
cnn_image_supersampling/
├── train.py, train_experimental.py        # Training scripts
├── diagnose_model.py, diagnose_model_experimental.py  # Testing scripts
├── test_model.py                          # Batch inference
├── checkpoints/
│   ├── best_model.pth                     # Standard model
│   └── model_experimental.pth             # Experimental model
├── diagnose/, diagnose_experimental/      # Test outputs
├── img/upscaled/                          # Batch upscale outputs
├── training_history*.png                  # Loss curves
└── resources/datasets/DIV2K/              # Training data
```

**Last Updated**: May 17, 2026

