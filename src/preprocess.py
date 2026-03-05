import argparse
import os
import numpy as np
import pandas as pd
from .utils import sliding_windows, split_indices, set_seed


def split_indices_by_group(
    group_ids: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int,
):
    unique = np.array(list(dict.fromkeys(group_ids.tolist())))
    n_groups = len(unique)
    if n_groups < 3:
        raise SystemExit(
            f"Need at least 3 groups for group-wise split; got {n_groups}. "
            "Use more groups or remove --group."
        )

    rng = np.random.RandomState(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)

    n_val = int(n_groups * val_frac)
    n_test = int(n_groups * test_frac)
    if val_frac > 0 and n_val == 0:
        n_val = 1
    if test_frac > 0 and n_test == 0:
        n_test = 1
    if n_val + n_test >= n_groups:
        # Keep at least one group for train.
        overflow = n_val + n_test - (n_groups - 1)
        while overflow > 0 and n_test > 0:
            n_test -= 1
            overflow -= 1
        while overflow > 0 and n_val > 0:
            n_val -= 1
            overflow -= 1

    val_groups = set(shuffled[:n_val].tolist())
    test_groups = set(shuffled[n_val:n_val + n_test].tolist())

    val_mask = np.isin(group_ids, list(val_groups))
    test_mask = np.isin(group_ids, list(test_groups))
    train_mask = ~(val_mask | test_mask)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, val_idx, test_idx


def standardize_by_train(
    X: np.ndarray,
    train_idx: np.ndarray,
    eps: float = 1e-6,
):
    x_train = X[train_idx].reshape(-1, X.shape[2]).astype(np.float64)
    feat_mean = x_train.mean(axis=0).astype(np.float32)
    feat_std = x_train.std(axis=0).astype(np.float32)
    feat_std = np.clip(feat_std, eps, None)
    X = X.astype(np.float64)
    X = (X - feat_mean[None, None, :]) / feat_std[None, None, :]
    return X.astype(np.float32), feat_mean, feat_std


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
    window_group_ids = []
    for gid, gdf in df.groupby(groups, sort=False):
        x = gdf[features].to_numpy(dtype=np.float32)
        windows = sliding_windows(x, window, stride)
        if windows.shape[0] == 0:
            continue
        X_list.append(windows)
        window_group_ids.extend([str(gid)] * windows.shape[0])

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
    if group_col:
        train_idx, val_idx, test_idx = split_indices_by_group(
            group_ids=np.asarray(window_group_ids),
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
        )
    else:
        train_idx, val_idx, test_idx = split_indices(len(X), val_frac, test_frac, seed)
    X, feat_mean, feat_std = standardize_by_train(X, train_idx)
    stats = {
        "mean": feat_mean,
        "std": feat_std,
        "features": np.array(features),
    }
    return X, y, train_idx, val_idx, test_idx, stats


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

    X, y, train_idx, val_idx, test_idx, stats = build_splits(
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
    stats_path = os.path.join(args.out, "stats.npz")
    np.savez_compressed(stats_path, mean=stats["mean"], std=stats["std"], features=stats["features"])
    print(f"Saved stats: {stats_path}")


if __name__ == "__main__":
    main()
