import argparse
import os
import numpy as np
import pandas as pd
from .utils import sliding_windows, split_indices, set_seed


def build_splits(
    df: pd.DataFrame,
    features: list[str],
    label_col: str | None,
    group_col: str | None,
    window: int,
    stride: int,
    label_mode: str,
    val_frac: float,
    test_frac: float,
    seed: int,
):
    groups = ["_all"] * len(df)
    if group_col:
        if group_col not in df.columns:
            raise SystemExit(f"Group column not found: {group_col}")
        groups = df[group_col].astype(str).tolist()

    X_list = []
    y_list = []
    for _, gdf in df.groupby(groups, sort=False):
        x = gdf[features].to_numpy(dtype=np.float32)
        windows = sliding_windows(x, window, stride)
        if windows.shape[0] == 0:
            continue
        X_list.append(windows)

        if label_col:
            y_raw = gdf[label_col].to_numpy(dtype=np.float32)
            y_win = sliding_windows(y_raw.reshape(-1, 1), window, stride).squeeze(-1)
            if label_mode == "last":
                y = y_win[:, -1]
            else:
                y = y_win.mean(axis=1)
            y_list.append(y)

    if not X_list:
        raise SystemExit("No windows created. Check window/stride or data length.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0) if y_list else None
    train_idx, val_idx, test_idx = split_indices(len(X), val_frac, test_frac, seed)
    return X, y, train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--features", required=True, help="Comma-separated feature columns")
    parser.add_argument("--label", required=False, default="", help="Label column for downstream")
    parser.add_argument("--group", default="", help="Optional group column (e.g., cycle or pack id)")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--label_mode", choices=["last", "mean"], default="last")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    df = pd.read_csv(args.data)

    features = [c.strip() for c in args.features.split(",") if c.strip()]
    if not features:
        raise SystemExit("No features provided")

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing}")

    label_col = args.label if args.label else None
    if label_col and label_col not in df.columns:
        raise SystemExit(f"Label column not found: {label_col}")

    X, y, train_idx, val_idx, test_idx = build_splits(
        df=df,
        features=features,
        label_col=label_col,
        group_col=args.group if args.group else None,
        window=args.window,
        stride=args.stride,
        label_mode=args.label_mode,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    def save_split(name, idx):
        path = os.path.join(args.out, f"{name}.npz")
        if y is None:
            np.savez_compressed(path, X=X[idx])
        else:
            np.savez_compressed(path, X=X[idx], y=y[idx])
        print(f"Saved {name}: {path} ({len(idx)} samples)")

    save_split("train", train_idx)
    save_split("val", val_idx)
    save_split("test", test_idx)


if __name__ == "__main__":
    main()
