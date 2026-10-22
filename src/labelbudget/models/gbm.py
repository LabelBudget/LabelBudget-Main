"""베이스라인 A·B (LightGBM)와 동일 예산 random_search. 스펙 02 §7.3.

모든 모델이 같은 n_trials를 쓴다. 탐색은 valid에서만 평가한다.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import numpy as np

from labelbudget.evaluation.metrics import classification_metrics, regression_metrics

# (종류, 하한, 상한). 저예산(100~300) 과적합 방지를 위해 min_child_samples·num_leaves를 넓게 둔다
LGBM_SPACE: dict[str, tuple] = {
    "n_estimators": ("int_log", 50, 1000),
    "learning_rate": ("log", 0.01, 0.3),
    "num_leaves": ("int_log", 4, 128),
    "min_child_samples": ("int_log", 2, 64),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.2, 1.0),
    "reg_lambda": ("log", 1e-3, 10.0),
}

# 탐색 지표: (이름, 클수록 좋은가)
SELECTION_METRIC = {"regression": ("rmse", False), "classification": ("auprc", True)}


def sample_params(space: dict[str, tuple], rng: np.random.Generator) -> dict[str, Any]:
    out = {}
    for name, (kind, lo, hi) in space.items():
        if kind == "float":
            out[name] = float(rng.uniform(lo, hi))
        elif kind == "log":
            out[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif kind == "int_log":
            out[name] = int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
        elif kind == "int":
            out[name] = int(rng.integers(lo, hi + 1))
        else:
            raise ValueError(f"알 수 없는 분포 '{kind}'")
    return out


def make_lgbm(task: str, params: dict, seed: int, n_jobs: int = 1):
    common = dict(random_state=seed, n_jobs=n_jobs, verbose=-1, deterministic=True, force_col_wise=True,
                  subsample_freq=1, **params)
    return lgb.LGBMRegressor(**common) if task == "regression" else lgb.LGBMClassifier(**common)


def predict(model, X, task: str) -> np.ndarray:
    return model.predict(X) if task == "regression" else model.predict_proba(X)[:, 1]


def score(task: str, y, pred) -> float:
    name, _ = SELECTION_METRIC[task]
    if task == "regression":
        return regression_metrics(y, pred)[name]
    return classification_metrics(y, pred, threshold=0.5)[name]


def search_trace(
    model_fn: Callable[[dict, int], Any],
    param_space: dict[str, tuple],
    X_tr, y_tr, X_va, y_va,
    *, n_trials: int, seed: int, task: str,
) -> list[tuple[dict, float]]:
    """시도 순서대로 (파라미터, valid 점수). 같은 seed면 같은 후보열이므로 앞 k개는 n_trials=k 탐색과 같다."""
    if n_trials < 1:
        raise ValueError("n_trials는 1 이상이어야 합니다")
    rng = np.random.default_rng(seed)
    trace = []
    for _ in range(n_trials):
        params = sample_params(param_space, rng)
        model = model_fn(params, seed).fit(X_tr, y_tr)
        trace.append((params, float(score(task, y_va, predict(model, X_va, task)))))
    return trace


def best_of(trace: list[tuple[dict, float]], k: int, task: str) -> tuple[dict, float]:
    """앞 k개 시도 중 valid 최고. NaN 점수는 건너뛴다."""
    _, higher_better = SELECTION_METRIC[task]
    valid = [(p, s) for p, s in trace[:k] if not np.isnan(s)]
    if not valid:
        raise RuntimeError("모든 시도가 NaN 점수를 냈습니다 (valid에 한 클래스만 있는지 확인)")
    pick = max if higher_better else min
    return pick(valid, key=lambda ps: ps[1])


def random_search(
    model_fn: Callable[[dict, int], Any],
    param_space: dict[str, tuple],
    X_tr, y_tr, X_va, y_va,
    *, n_trials: int, seed: int, task: str,
) -> tuple[dict, float]:
    """valid 점수가 가장 좋은 파라미터와 그 점수."""
    trace = search_trace(model_fn, param_space, X_tr, y_tr, X_va, y_va, n_trials=n_trials, seed=seed, task=task)
    return best_of(trace, n_trials, task)
