# E. coli Growth Stage Detection and Classification

Deep learning approaches for detecting and classifying *E. coli* bacteria into three growth stages using microscopy images.

**Classes:** rod (single cells), dividing (binary fission), microcolony (4+ cell clusters)

**Dataset:** DeepBacs E. coli - 100 train + 60 test grayscale 256x256 images

## Project Structure

```
├── data/
│   ├── raw/image_patches/          # Original DeepBacs dataset
│   │   ├── train_images/
│   │   ├── train_annotations/
│   │   ├── test_images/
│   │   └── test_annotations/
│   └── Augmented/                  # Augmented dataset
│
├── models/
│   ├── CNN-sliding-window/         # Sliding window CNN classifier
│   ├── U-Net/                      # U-Net semantic segmentation
│   └── YOLOv8/                     # YOLOv8 object detection
│
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Models

### 1. CNN Sliding Window Classifier

A custom 4-layer CNN that classifies 64x64 patches extracted via sliding window.

**Run from `models/CNN-sliding-window/` directory:**

```bash
cd models/CNN-sliding-window

# Step 1: Prepare patch dataset
python scripts/prepare_cnn_data.py

# Step 2: Train the model
python scripts/train_cnn.py --epochs 100 --batch-size 32
```

**Results:** Best F1 Score: 0.8277

### 2. U-Net Semantic Segmentation

Standard U-Net encoder-decoder architecture for pixel-wise segmentation.

**Run from `models/U-Net/` directory:**

```bash
cd models/U-Net

# Step 1: Prepare segmentation masks
python scripts/prepare_masks.py --source ../../data/raw/image_patches --output ../../data/unet_original

# Step 2: Train the model
python scripts/train_unet.py --data-dir ../../data/unet_original --epochs 100
```

**Results:** Best mIoU (excluding background): 0.4013

### 3. YOLOv8 Object Detection

Fine-tuned YOLOv8 for bacterial detection.

**Run the Jupyter notebooks in `models/YOLOv8/`:**

1. `convert_txt.ipynb` - Convert annotations to YOLO format
2. `yolov8.ipynb` - Train and evaluate YOLOv8

**Results:** Best mAP@50: 0.954

## Results Summary

| Model | Primary Metric | Parameters |
|-------|---------------|------------|
| CNN Sliding Window | F1: 82.77% | ~254K |
| U-Net | mIoU: 40.13% | ~31M |
| YOLOv8n | mAP@50: 95.4% | ~11M |

## Team

- Alan Guo (weihaog1@uci.edu)
- Mark Xu (markyx@uci.edu)
- William Ma (qijunm@uci.edu)

CS 184A Fall 2025, UC Irvine
