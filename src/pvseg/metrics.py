from __future__ import annotations

import numpy as np
from PIL import Image


def load_binary_mask(path: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if shape is not None and image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def load_probability(path: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path)
    if shape is not None and image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return array / (65535.0 if array.max() > 255 else 255.0)


def dice(pred: np.ndarray, target: np.ndarray, empty_agreement: float = 0.0) -> float:
    pred_sum = int(pred.sum())
    target_sum = int(target.sum())
    if pred_sum == 0 and target_sum == 0:
        return float(empty_agreement)
    return float(2 * np.logical_and(pred, target).sum() / max(pred_sum + target_sum, 1))


def iou(pred: np.ndarray, target: np.ndarray, empty_agreement: float = 0.0) -> float:
    union = int(np.logical_or(pred, target).sum())
    if union == 0:
        return float(empty_agreement)
    return float(np.logical_and(pred, target).sum() / union)
