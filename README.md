# EVBattery – Self-Supervised Battery Degradation Representations

This project implements a **self-supervised representation learning** pipeline for EV battery diagnostics using the Kaggle dataset:
`drtawfikrrahman/deep-learning-ev-battery-pack-diagnostics-sdg-7`.

## Goals
- Learn **time-series representations** from unlabeled signals via contrastive learning.
- Evaluate **downstream SOH prediction** with limited labels.
- Improve SOH prediction quality with a reproducible, leakage-safe training pipeline.

## Final Result

Current best-performing configuration:
- Features: `Voltage,Current,Temperature,ICA`
- Group-wise split: `CellID`
- Window / stride: `256 / 8`
- Label aggregation: `last`
- SSL pretraining: `20` epochs, `crop=224`, `lr=5e-4`
- Downstream head: MLP with `hidden=128`, `dropout=0.2`
- Fine-tuning: `--finetune all`
- Label fraction: `0.1`

Best reproduced metrics:

| Config | Val RMSE | Val MAE | Test RMSE | Test MAE | Test R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `V/I/T` + `SSL5` + `finetune=last` | 0.1207 | 0.1019 | 0.1214 | 0.1025 | 0.0973 |
| `V/I/T/ICA` + `SSL20` + `finetune=all` | 0.0235 | 0.0183 | 0.0243 | 0.0198 | 0.9637 |

Impact:
- Test RMSE improved from `0.1214` to `0.0243` under the stricter `CellID` split.
- That is about a `79.96%` reduction in test RMSE.
- Re-running the same configuration with the same seed reproduced the same validation and test metrics exactly.

## What Changed

The main improvements were:
- **Leakage-safe evaluation**: train/val/test are split by `CellID`, so windows from the same cell do not appear in multiple splits.
- **Train-only normalization**: each feature is standardized using only train-split statistics and saved in `stats.npz`.
- **Better encoder pooling**: the encoder now uses attention pooling instead of simple mean pooling.
- **Stronger downstream head**: a small MLP replaced the original linear head.
- **End-to-end fine-tuning**: the best result came from `--finetune all`, not freezing the encoder.
- **Feature selection by ablation**: `ICA` helped, while adding `DVA` made performance worse in the final tuned setup.

## Ablation Summary

Under the same tuned pipeline:

| Features | Test RMSE | Test MAE | Test R2 |
| --- | ---: | ---: | ---: |
| `Voltage,Current,Temperature,ICA,DVA` | 0.0308 | 0.0249 | 0.9420 |
| `Voltage,Current,Temperature,ICA` | 0.0243 | 0.0198 | 0.9637 |
| `Voltage,Current,Temperature,DVA` | 0.0292 | 0.0232 | 0.9479 |

Takeaway:
- `ICA` is useful.
- `DVA` is not helpful in the final tuned pipeline.
- The recommended final feature set is `Voltage,Current,Temperature,ICA`.

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
- Use `--group CellID` to enforce group-wise splitting and avoid leakage across train/val/test.
- Use `--label_mode last|mean` to control how the label is aggregated per window.
- Use `--seed` to make the train/val/test split deterministic.
- Train-split normalization statistics are exported to `stats.npz`.

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
- `data/processed/stats.npz`
- Each `.npz` contains `X` (and `y` if a label was provided)

## 4) Self-supervised pretraining

Notes:
- `--seed` makes augmentation and batching deterministic.
- The current encoder uses attention pooling.

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
- `--save_best` writes the best downstream MLP by val RMSE.
- `--finetune` controls whether the encoder is frozen, partially tuned, or fully tuned.
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
- `runs/downstream/best_encoder.pt` when fine-tuning is enabled
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

## 7) Reproduce The Best Run

This is the tuned recipe that produced the best result in the repo:

```bash
python3 -m src.preprocess \
  --data data/raw/dataset.csv \
  --out data/processed_w256s8_last_vitica \
  --features "Voltage,Current,Temperature,ICA" \
  --label "SoH" \
  --group "CellID" \
  --window 256 \
  --stride 8 \
  --label_mode last \
  --seed 42

python3 -m src.ssl_pretrain \
  --data data/processed_w256s8_last_vitica/train.npz \
  --out runs/ssl_w256s8_last_vitica_e20 \
  --epochs 20 \
  --batch 128 \
  --lr 5e-4 \
  --crop 224 \
  --proj 128 \
  --seed 42

python3 -m src.downstream \
  --data data/processed_w256s8_last_vitica/train.npz \
  --val data/processed_w256s8_last_vitica/val.npz \
  --ckpt runs/ssl_w256s8_last_vitica_e20/encoder.pt \
  --label_frac 0.1 \
  --epochs 80 \
  --batch 128 \
  --lr 3e-4 \
  --finetune all \
  --head_hidden 128 \
  --dropout 0.2 \
  --seed 42 \
  --out runs/downstream_w256s8_last_vitica_e20_ftall \
  --save_best \
  --plot \
  --curve \
  --save_pred
```

Reference result files:
- `runs/downstream_w256s8_last_vitica_e20_ftall/metrics.json`
- `runs/downstream_w256s8_last_vitica_e20_ftall/test_metrics.json`
- `runs/ablation_summary.csv`

## 8) One-command pipeline

Run everything (inspect → preprocess → SSL pretrain → downstream eval → sweep):

```bash
bash run_all.sh
```

`run_all.sh` is a convenience baseline pipeline. It does **not** reproduce the best tuned result above.

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
- The current recommended production setting is `Voltage,Current,Temperature,ICA` with the tuned recipe shown above.

## Directory layout
```
EVBattery/
  src/
  config/
  data/
  runs/
```
