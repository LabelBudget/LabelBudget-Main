"""시드 쌍 Wilcoxon 검정. 스펙 02 §9."""
from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy import stats

MIN_PAIRS = 5


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    """시드별 쌍 비교 (a - b). return {statistic, p_value, median_diff, mean_diff, n_pairs}"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"쌍의 개수가 다릅니다: {a.shape} vs {b.shape}")
    n = len(a)
    if n < MIN_PAIRS:
        warnings.warn(f"쌍이 {n}개뿐입니다 (<{MIN_PAIRS}). 검정력이 없습니다", stacklevel=2)
    d = a - b
    if n == 0 or np.allclose(d, 0):
        stat, p = float("nan"), 1.0
    else:
        res = stats.wilcoxon(a, b)
        stat, p = float(res.statistic), float(res.pvalue)
    return {"statistic": stat, "p_value": p, "median_diff": float(np.median(d)) if n else float("nan"),
            "mean_diff": float(d.mean()) if n else float("nan"), "n_pairs": n}


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni 보정 p값 (입력 순서 유지)."""
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj
