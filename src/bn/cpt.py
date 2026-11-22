"""CPT construction helpers."""

from __future__ import annotations

import numpy as np


def bernoulli_table(p_present: float) -> np.ndarray:
    p = float(np.clip(p_present, 1e-4, 1 - 1e-4))
    # rows: state absent/present
    return np.array([1 - p, p], dtype=float)


def noisy_or_severe_table(n_parents: int, leak: float = 0.05, p_cause: float = 0.45) -> np.ndarray:
    """P(severe=yes | parent bitvector) via noisy-OR.

    CPT indexed by parent configuration integer 0..2^n-1 (bit i = parent i present).
    Returns array shape (2^n, 2) for [no, yes].
    """
    n_cfg = 1 << n_parents
    table = np.zeros((n_cfg, 2), dtype=float)
    for cfg in range(n_cfg):
        q = 1.0 - leak
        for i in range(n_parents):
            if cfg & (1 << i):
                q *= 1.0 - p_cause
        p_yes = 1.0 - q
        table[cfg] = [1.0 - p_yes, p_yes]
    return table
