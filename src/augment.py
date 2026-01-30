import numpy as np


def jitter(x: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    return x + np.random.normal(0, sigma, size=x.shape).astype(x.dtype)


def scaling(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    factor = np.random.normal(1.0, sigma, size=(x.shape[0], 1)).astype(x.dtype)
    return x * factor


def random_crop(x: np.ndarray, crop_size: int) -> np.ndarray:
    if x.shape[0] <= crop_size:
        return x
    start = np.random.randint(0, x.shape[0] - crop_size + 1)
    return x[start:start + crop_size]


def augment_pair(x: np.ndarray, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    a = random_crop(x, crop_size)
    b = random_crop(x, crop_size)
    a = scaling(jitter(a))
    b = scaling(jitter(b))
    return a, b
