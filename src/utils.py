import numpy as np
import random


def set_seed(seed: int) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def sliding_windows(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    if x.shape[0] < window:
        return np.empty((0, window, x.shape[1]), dtype=x.dtype)
    num = 1 + (x.shape[0] - window) // stride
    out = np.empty((num, window, x.shape[1]), dtype=x.dtype)
    for i in range(num):
        start = i * stride
        out[i] = x[start:start + window]
    return out


def split_indices(n: int, val_frac: float, test_frac: float, seed: int = 42):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return train_idx, val_idx, test_idx
