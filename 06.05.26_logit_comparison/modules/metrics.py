from __future__ import annotations

import numpy as np


def acc_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.argmax(logits, axis=1) == y_true))


def mse_onehot_from_logits(logits: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    labels_onehot = np.zeros((y_true.size, num_classes), dtype=np.float32)
    labels_onehot[np.arange(y_true.size), y_true] = 1.0
    return float(np.mean((logits - labels_onehot) ** 2))
