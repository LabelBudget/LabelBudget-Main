import numpy as np
import pytest
from scipy import stats as sps

from labelbudget.evaluation.metrics import choose_threshold, classification_metrics, regression_metrics
from labelbudget.evaluation.stats import holm, paired_wilcoxon


def test_regression_metrics_r2_is_not_pearson_squared():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    p = y * 2 + 5  # 완전 상관이지만 크게 빗나감
    m = regression_metrics(y, p)
    assert m["pearson"] == pytest.approx(1.0)
    assert m["r2"] < 0
    assert m["rmse"] == pytest.approx(np.sqrt(((p - y) ** 2).mean()))


def test_classification_metrics_counts():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.2, 0.1, 0.3])
    m = classification_metrics(y, p, threshold=0.5)
    assert m["ppv"] == pytest.approx(0.5) and m["npv"] == pytest.approx(0.75)
    assert m["n_positive"] == 2 and m["low_positive_flag"]
    assert m["auroc"] == pytest.approx(7 / 8)


def test_choose_threshold_on_valid():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.35, 0.9])
    assert choose_threshold(y, p) == pytest.approx(0.35)


def test_paired_wilcoxon_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=10)
    b = a - 0.5 + rng.normal(scale=0.1, size=10)
    r = paired_wilcoxon(a, b)
    assert r["p_value"] == pytest.approx(sps.wilcoxon(a, b).pvalue)
    assert r["n_pairs"] == 10 and r["median_diff"] > 0


def test_paired_wilcoxon_warns_few_pairs():
    with pytest.warns(UserWarning, match="검정력"):
        paired_wilcoxon([1, 2, 3], [0, 1, 2.5])


def test_holm():
    assert holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
