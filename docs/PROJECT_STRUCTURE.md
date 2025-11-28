# Project Structure

## Directory Layout

```
E-coli-growth-stages-detection-and-classification/
├── configs/
│   └── config.yaml          # Hyperparameters and paths
├── data/
│   ├── raw/                  # Original dataset (download here)
│   ├── processed/            # Converted to YOLO format
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── augmented/            # Offline augmented data
│   └── dataset.yaml          # YOLO dataset config
├── docs/
│   └── PROJECT_STRUCTURE.md
├── notebooks/                # Jupyter notebooks for EDA
├── scripts/
│   ├── train_cnn.py          # Train custom CNN
│   └── train_yolo.py         # Train YOLOv8
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── augmentation.py   # Albumentations pipelines
│   │   ├── convert_annotations.py  # XML to YOLO converter
│   │   └── dataset.py        # PyTorch Dataset classes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py            # Custom CNN architecture
│   │   └── detector.py       # Sliding window detector
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py        # Seed, config, device
│       └── metrics.py        # mAP, F1, confusion matrix
├── weights/                  # Saved model checkpoints
├── runs/                     # YOLO training outputs
├── .gitignore
└── requirements.txt
```

## Design Decisions

### 1. Separation of Source Code and Scripts

**Why**: `src/` contains reusable modules. `scripts/` contains entry points for running experiments.

- Modules in `src/` can be imported anywhere
- Scripts are standalone executables
- Avoids circular imports
- Makes testing easier

**When to use this pattern**: Any project with more than 2-3 files. Keeps code organized as complexity grows.

### 2. Config File Instead of Hardcoded Values

**Why**: `configs/config.yaml` centralizes all hyperparameters.

- Change settings without editing code
- Easy to track experiments (save config with results)
- Compare runs by diffing configs
- Share settings with teammates

**Example**:
```yaml
cnn:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
```

### 3. Separate Data Directories

**Why**: `raw/`, `processed/`, `augmented/` keep data stages isolated.

- `raw/` contains untouched original data
- `processed/` contains format-converted data ready for training
- `augmented/` contains offline-generated augmented samples
- Easy to regenerate any stage without affecting others
- `.gitignore` excludes these to avoid committing large files

### 4. Albumentations for Augmentation

**Why**: Better than torchvision for object detection.

- Automatically transforms bounding boxes with images
- Faster (OpenCV backend)
- One pipeline handles both image and labels
- Rich set of augmentations

**Example**:
```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
], bbox_params=A.BboxParams(format="yolo"))
```

### 5. Two Dataset Classes

**Why**: Different approaches need different data formats.

- `EcoliDataset`: Returns full images with bounding boxes (for YOLO-style detection)
- `EcoliPatchDataset`: Returns cropped patches with class labels (for CNN classifier)

This separation keeps each class simple and focused.

### 6. Modular Model Code

**Why**: `models/` contains only architecture definitions.

- `cnn.py`: Defines the network layers
- `detector.py`: Wraps the CNN with sliding window logic

Training logic stays in scripts, not in model files. This lets you:
- Reuse models in different training setups
- Test models without running training
- Swap models easily

### 7. Utils for Common Operations

**Why**: Functions used everywhere go in `utils/`.

- `set_seed()`: Reproducibility
- `load_config()`: Config loading
- `compute_metrics()`: Evaluation

Avoids duplicating code across scripts.

## Data Flow

```
1. Download dataset
   └── data/raw/

2. Convert annotations
   python -m src.data.convert_annotations --input data/raw --output data/processed/labels
   └── data/processed/labels/

3. Split images into train/val/test
   └── data/processed/images/

4. (Optional) Generate augmented data
   └── data/augmented/

5. Train models
   python scripts/train_cnn.py
   python scripts/train_yolo.py
   └── weights/, runs/

6. Evaluate
   └── Results in console/logs
```

## File Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | lowercase, underscores | `convert_annotations.py` |
| Classes | PascalCase | `EcoliDataset` |
| Functions | lowercase, underscores | `compute_iou()` |
| Constants | UPPERCASE | `CLASS_TO_ID` |
| Configs | lowercase, hyphens ok | `config.yaml` |

## Adding New Components

### New model

1. Create `src/models/new_model.py`
2. Add class to `src/models/__init__.py`
3. Create training script in `scripts/`

### New augmentation

1. Add function to `src/data/augmentation.py`
2. Update transforms in dataset or script

### New metric

1. Add function to `src/utils/metrics.py`
2. Import in evaluation scripts

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Convert annotations
python -m src.data.convert_annotations --input data/raw --output data/processed/labels

# Train CNN
python scripts/train_cnn.py --epochs 100 --batch-size 32

# Train YOLO
python scripts/train_yolo.py --epochs 100 --batch 16
```

