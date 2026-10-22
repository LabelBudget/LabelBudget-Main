"""회귀·분류 지표. 스펙 02 §8.

- 회귀 주 지표 RMSE. R²는 1 - SS_res/SS_tot 로 직접 계산 (Pearson r² 아님)
- 분류 주 지표 AUPRC. PPV/NPV 임계값은 valid에서 정하고 test에 그대로 적용
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

MIN_TEST_POSITIVE = 20


def regression_metrics(y_true, y_pred) -> dict:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    resid = y - p
    ss_tot = float(((y - y.mean()) ** 2).sum())
    varied = np.ptp(y) > 0 and np.ptp(p) > 0  # 상수 예측이면 상관계수는 정의되지 않는다
    return {
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "mae": float(np.abs(resid).mean()),
        "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "spearman": float(stats.spearmanr(y, p).statistic) if varied else float("nan"),
        "pearson": float(stats.pearsonr(y, p).statistic) if varied else float("nan"),
        "n": int(len(y)),
    }


def choose_threshold(y_valid, p_valid) -> float:
    """valid에서 MCC를 최대화하는 임계값. 동률이면 작은 값."""
    y, p = np.asarray(y_valid, int), np.asarray(p_valid, float)
    if len(np.unique(y)) < 2:
        return 0.5
    best_t, best = 0.5, -np.inf
    for t in np.unique(p):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best:
            best_t, best = float(t), m
    return best_t


def classification_metrics(y_true, p_score, threshold: float) -> dict:
    y, p = np.asarray(y_true, int), np.asarray(p_score, float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_pos = int(y.sum())
    both = n_pos > 0 and n_pos < len(y)
    return {
        "auprc": float(average_precision_score(y, p)) if both else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else float("nan"),
        "threshold": float(threshold),
        "n": int(len(y)),
        "n_positive": n_pos,
        "imbalance_ratio": (len(y) - n_pos) / n_pos if n_pos else float("inf"),
        "low_positive_flag": n_pos < MIN_TEST_POSITIVE,
    }
