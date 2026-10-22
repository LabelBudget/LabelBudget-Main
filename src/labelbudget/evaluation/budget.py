"""라벨 예산 서브샘플링. 스펙 02 §10.

서브샘플링 시드와 분할 시드는 분리한다. 테스트셋은 절대 건드리지 않는다.
"""
from __future__ import annotations

import warnings

import numpy as np

BUDGETS_REGRESSION = [100, 300, 1000, 3000, None]   # None = 전체
BUDGETS_CLASSIFICATION = [300, 1000, 3000, None]     # 100 제외 (§1.4)
MIN_POSITIVE = 20


class LowPositiveWarning(UserWarning):
    pass


def subsample_train(y_train, *, n_labels: int | None, seed: int, task: str, stratify: bool = True) -> np.ndarray:
    """학습셋 내 위치 인덱스를 n_labels개로 줄인다 (정렬된 배열).

    n_labels가 None이거나 학습셋 이상이면 전체를 반환한다.
    분류는 양성 비율을 유지하는 층화 샘플링 (양성 수 = round(n_labels × 양성률), 최소 1).
    """
    y = np.asarray(y_train)
    n = len(y)
    if n_labels is None or n_labels >= n:
        idx = np.arange(n)
    elif n_labels <= 0:
        raise ValueError("n_labels는 양수여야 합니다")
    else:
        rng = np.random.default_rng(seed)
        if task == "classification" and stratify:
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            k_pos = int(round(n_labels * len(pos) / n))
            k_pos = min(max(k_pos, 1 if len(pos) else 0), len(pos), n_labels)
            k_neg = min(n_labels - k_pos, len(neg))
            idx = np.concatenate([rng.choice(pos, k_pos, replace=False), rng.choice(neg, k_neg, replace=False)])
        else:
            idx = rng.choice(n, n_labels, replace=False)
        idx = np.sort(idx)

    if task == "classification":
        n_pos = int((y[idx] == 1).sum())
        if n_pos < MIN_POSITIVE:
            warnings.warn(f"예산 {n_labels}: 양성 {n_pos}개 (<{MIN_POSITIVE}). 이 조합은 신뢰할 수 없습니다",
                          LowPositiveWarning, stacklevel=2)
    return idx


def budget_info(y_train, idx: np.ndarray, n_labels: int | None, task: str) -> dict:
    """결과 테이블 기록용."""
    y = np.asarray(y_train)[idx]
    info = {"n_labels_requested": n_labels, "n_train_used": int(len(idx)),
            "used_full_train": n_labels is None or n_labels >= len(y_train)}
    if task == "classification":
        n_pos = int((y == 1).sum())
        info.update(n_train_positive=n_pos, low_positive_flag=n_pos < MIN_POSITIVE)
    return info
