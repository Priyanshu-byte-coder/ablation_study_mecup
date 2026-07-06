# CON-SOL-E 5.0 — Ablation Study
DINOv2 ViT-B/14 + Custom Dense FPN-UNet Decoder  
4-class surface defect segmentation (Background / Dust / RunDown / Scratch)

---

## Architecture
- **Encoder**: DINOv2 `dinov2_vitb14` (~86M params, frozen during ablation)
- **Decoder**: Custom Dense FPN-UNet, skip_layers=[3,7,11], channels=[256,128,64]
- **Loss**: 0.5×DiceLoss + 0.5×FocalLoss (α=0.25, γ=2.0), class_weights=[0.3, 3.0, 2.0, 2.0]
- **Optimizer**: AdamW, lr=5e-5, weight_decay=0.01

---

## Hardware Requirements
- GPU: ≥6GB VRAM (tested on RTX 3050 6GB and Kaggle T4)
- RAM: ≥16GB
- Storage: ~3GB free (model weights + dataset)

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Dataset: **Paint_Defect v7** from Roboflow  
URL: https://universe.roboflow.com/shyam-sojitra-i8tgx/paint_defect-hncxs  
License: CC BY 4.0

Download in **YOLOv8 format** and place at:
```
data/data/
├── train/
│   ├── images/   (1081 images)
│   └── labels/
├── valid/
│   ├── images/   (56 images)
│   └── labels/
└── test/
    ├── images/   (57 images)
    └── labels/
```

> **Note:** Format must be YOLO polygon (segmentation), not bounding box. Select "YOLOv8" export with segmentation annotations.

---

## Running Ablation Study

### Full 100-epoch ablation (all variants)
```bash
python run_ablation.py --mode full --epochs 100 --output ablation_results_100ep.json
```

### Specific variants only
```bash
python run_ablation.py --mode full --epochs 100 --variants A5_dice_only A7_no_class_weights A8_small_decoder
```

### Scale (skip-layer) variants
```bash
python run_ablation_scale.py --epochs 100 --output ablation_results_scale_100ep.json
```

### Quick test run (15 epochs)
```bash
python run_ablation.py --mode quick --epochs 15 --output ablation_results_quick.json
```

---

## Ablation Variants

| ID | Name | What changes |
|----|------|-------------|
| A1_fresh_local | Full model (baseline) | All 3 skip layers [3,7,11] |
| A2_single_scale | Single scale | skip_layers=[11] only |
| A3_two_scale | Two scale | skip_layers=[7,11] |
| A5_dice_only | Dice loss only | No focal loss |
| A7_no_class_weights | No class weights | Uniform loss weighting |
| A8_small_decoder | Small decoder | channels=[128,64,32] |

---

## Current Results (15 epochs, RTX 3050)

| Variant | mIoU | Dust IoU | RunDown IoU | Scratch IoU |
|---------|------|----------|-------------|-------------|
| A1_fresh_local (baseline) | 45.2% | 57.7% | 26.6% | 30.3% |
| A2_single_scale | 44.4% | 58.8% | 17.9% | 32.7% |
| A3_two_scale | 46.0% | 57.8% | 26.8% | 32.7% |
| A5_dice_only | 43.6% | 53.8% | 18.7% | 27.7% |
| A7_no_class_weights | 45.4% | 54.9% | 22.2% | 29.2% |
| A8_small_decoder | 38.7% | 52.4% | 10.1% | 24.8% |

> Full model trained on proprietary Mitsubishi dataset (not included): **81.1% mIoU** at epoch 68.

---

## View Results After Training
```bash
python run_ablation.py --mode eval_only --output ablation_results_100ep.json
```

---

## Key Bug Fixes (already applied)
- `models/encoder.py`: Variable skip_layers correctly assigns keys from END of list; missing keys filled with zero tensors
- `data/dataset.py`: Windows case-insensitive glob deduplication fix
- `evaluate.py`: 4-class names fix (`Background/Dust/RunDown/Scratch`), handles `valid/` directory
