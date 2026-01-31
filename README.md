# EVBattery – Self-Supervised Battery Degradation Representations

This project implements a **self-supervised representation learning** pipeline for EV battery diagnostics using the Kaggle dataset:
`drtawfikrrahman/deep-learning-ev-battery-pack-diagnostics-sdg-7`.

## Goals
- Learn **time-series representations** from unlabeled signals via contrastive learning.
- Evaluate **downstream SOH prediction** with limited labels.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Download dataset

You must provide the file name inside the Kaggle dataset (example: `EV_Battery.csv`).

```bash
python3 -m src.download --out data/raw --file "cell_level_dataset.csv"
```

## 2) Inspect columns

```bash
python3 -m src.inspect --data data/raw/dataset.csv
```

## 3) Preprocess into windows

Notes:
- Use `--group CellID` to avoid mixing different cells in a single window.
- Use `--label_mode last|mean` to control how the label is aggregated per window.
- Use `--seed` to make the train/val/test split deterministic.

```bash
python3 -m src.preprocess \
  --data data/raw/dataset.csv \
  --out data/processed \
  --features "Voltage,Current,Temperature" \
  --label "SoH" \
  --group "CellID" \
  --window 128 \
  --stride 16 \
  --seed 42
```

Outputs:
- `data/processed/train.npz`, `val.npz`, `test.npz`
- Each `.npz` contains `X` (and `y` if a label was provided)

## 4) Self-supervised pretraining

Notes:
- `--seed` makes augmentation and batching deterministic.

```bash
python3 -m src.ssl_pretrain \
  --data data/processed/train.npz \
  --out runs/ssl \
  --seed 42
```

Outputs:
- `runs/ssl/encoder.pt`

## 5) Downstream evaluation (few labels)

Notes:
- `--save_best` writes the best linear regressor by val RMSE.
- `--plot` saves a scatter plot of true vs predicted.
- `--curve` saves a line plot of true vs predicted by index.
- `--save_pred` saves a CSV with per-sample predictions.

```bash
python3 -m src.downstream \
  --data data/processed/train.npz \
  --val data/processed/val.npz \
  --ckpt runs/ssl/encoder.pt \
  --label_frac 0.1 \
  --seed 42 \
  --out runs/downstream \
  --save_best \
  --plot \
  --curve \
  --save_pred
```

Outputs (when flags are enabled):
- `runs/downstream/best_reg.pt`
- `runs/downstream/metrics.json`
- `runs/downstream/pred_vs_true.png`
- `runs/downstream/pred_vs_true_curve.png`
- `runs/downstream/val_predictions.csv`

## 6) Hyperparameter sweep (window/stride/label_mode/label_frac)

Notes:
- This runs multiple trials on the raw CSV and reports val RMSE.
- Results are written to a single CSV for easy comparison.

```bash
python3 -m src.tune \
  --data data/raw/dataset.csv \
  --ckpt runs/ssl/encoder.pt \
  --features "Voltage,Current,Temperature" \
  --label "SoH" \
  --group "CellID" \
  --windows "64,128,256" \
  --strides "8,16" \
  --label_modes "last,mean" \
  --label_fracs "0.05,0.1" \
  --out runs/tune
```

Outputs:
- `runs/tune/results.csv`
- `runs/tune/trial_###/best_reg.pt`
- `runs/tune/trial_###/metrics.json`

## 7) One-command pipeline

Run everything (inspect → preprocess → SSL pretrain → downstream eval → sweep):

```bash
bash run_all.sh
```

Override defaults (ordered arguments):

```bash
bash run_all.sh <raw_csv> <features> <label> <group> <window> <stride> <seed> <label_frac>
```

Example:

```bash
bash run_all.sh data/raw/dataset.csv "Voltage,Current,Temperature" SoH CellID 128 16 42 0.1
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
