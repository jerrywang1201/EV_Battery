# EVBattery – Self-Supervised Battery Degradation Representations

This project implements a **self-supervised representation learning** pipeline for EV battery diagnostics using the Kaggle dataset:
`drtawfikrrahman/deep-learning-ev-battery-pack-diagnostics-sdg-7`.

## Goals
- Learn **time-series representations** from unlabeled signals via contrastive learning.
- Evaluate **downstream SOH prediction** with limited labels.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Download dataset

You must provide the file name inside the Kaggle dataset (example: `EV_Battery.csv`).

```bash
python -m src.download --out data/raw --file \"<dataset-file>.csv\"
```

## 2) Inspect columns

```bash
python -m src.inspect --data data/raw/dataset.csv
```

## 3) Preprocess into windows

```bash
python -m src.preprocess \
  --data data/raw/dataset.csv \
  --out data/processed \
  --features "Voltage,Current,Temperature" \
  --label "SOH" \
  --window 128 \
  --stride 16
```

## 4) Self-supervised pretraining

```bash
python -m src.ssl_pretrain \
  --data data/processed/train.npz \
  --out runs/ssl
```

## 5) Downstream evaluation (few labels)

```bash
python -m src.downstream \
  --data data/processed/train.npz \
  --val data/processed/val.npz \
  --ckpt runs/ssl/encoder.pt \
  --label_frac 0.1
```

## Notes
- This repo assumes **time-series data** with numeric columns.
- Use `src.inspect` to locate the correct label column (e.g., `SOH` or similar).
- The code is modular: you can replace augmentations, encoder, and loss.

## Directory layout
```
EVBattery/
  src/
  config/
  data/
  runs/
```
