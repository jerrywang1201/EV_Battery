import argparse
import csv
import itertools
import os
import numpy as np
import pandas as pd
import torch
from .preprocess import build_splits
from .model import Encoder1D
from .downstream import train_regressor
from .utils import set_seed


def _parse_list(value: str, cast):
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [cast(v) for v in items]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Raw CSV")
    parser.add_argument("--ckpt", required=True, help="Pretrained encoder checkpoint")
    parser.add_argument("--features", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--group", default="")
    parser.add_argument("--windows", default="128")
    parser.add_argument("--strides", default="16")
    parser.add_argument("--label_modes", default="last")
    parser.add_argument("--label_fracs", default="0.1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/tune")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.data)
    features = [c.strip() for c in args.features.split(",") if c.strip()]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing}")
    label_col = args.label
    if label_col not in df.columns:
        raise SystemExit(f"Label column not found: {label_col}")

    windows = _parse_list(args.windows, int)
    strides = _parse_list(args.strides, int)
    label_modes = _parse_list(args.label_modes, str)
    label_fracs = _parse_list(args.label_fracs, float)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Encoder1D(in_ch=len(features), out_dim=128).to(device)
    enc.load_state_dict(torch.load(args.ckpt, map_location=device))
    enc.eval()

    results = []
    trial_id = 0
    for window, stride, label_mode, label_frac in itertools.product(
        windows, strides, label_modes, label_fracs
    ):
        trial_id += 1
        trial_seed = args.seed + trial_id
        set_seed(trial_seed)

        X, y, train_idx, val_idx, _ = build_splits(
            df=df,
            features=features,
            label_col=label_col,
            group_col=args.group if args.group else None,
            window=window,
            stride=stride,
            label_mode=label_mode,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=trial_seed,
        )

        Xtr = X[train_idx]
        ytr = y[train_idx]
        Xv = X[val_idx]
        yv = y[val_idx]

        out_dir = os.path.join(args.out, f"trial_{trial_id:03d}")
        best, _ = train_regressor(
            enc=enc,
            X=Xtr,
            y=ytr,
            Xv=Xv,
            yv=yv,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            label_frac=label_frac,
            seed=trial_seed,
            out_dir=out_dir,
            save_best=True,
        )

        results.append(
            {
                "trial": trial_id,
                "window": window,
                "stride": stride,
                "label_mode": label_mode,
                "label_frac": label_frac,
                "val_mae": best["mae"],
                "val_rmse": best["rmse"],
                "best_epoch": best["epoch"],
                "train_samples": int(len(Xtr)),
                "val_samples": int(len(Xv)),
            }
        )

    csv_path = os.path.join(args.out, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best = sorted(results, key=lambda r: r["val_rmse"])[0]
    print(f"Saved results: {csv_path}")
    print(
        "Best config:",
        f"window={best['window']}",
        f"stride={best['stride']}",
        f"label_mode={best['label_mode']}",
        f"label_frac={best['label_frac']}",
        f"rmse={best['val_rmse']:.4f}",
    )


if __name__ == "__main__":
    main()
