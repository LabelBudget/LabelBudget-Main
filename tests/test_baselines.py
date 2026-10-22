import numpy as np
import pandas as pd
import pytest

from labelbudget.evaluation.runner import FeatureStore, run_grid
from labelbudget.featurize.descriptors import DescriptorSelector, compute_descriptors
from labelbudget.featurize.fingerprints import ecfp4
from labelbudget.models.gbm import LGBM_SPACE, make_lgbm, random_search, sample_params

SMILES = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "c1ccncc1", "Cc1ccncc1",
          "C1CCCCC1", "CC1CCCCC1", "CCOC(=O)C", "CC(=O)O", "CCN", "CCCN", "c1ccc2ccccc2c1", "Oc1ccccc1"]


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    smiles = [SMILES[i % len(SMILES)] + "C" * (i // len(SMILES)) for i in range(160)]
    df = pd.DataFrame({"smiles": smiles, "inchikey": [f"K{i:04d}-X-N" for i in range(160)]})
    df["label"] = [s.count("C") * 0.3 + rng.normal(scale=0.1) for s in smiles]
    return df


def test_ecfp4_shape_and_bits():
    X = ecfp4(["CCO", "c1ccccc1"])
    assert X.shape == (2, 2048) and X.dtype == np.uint8 and set(np.unique(X)) <= {0, 1}


def test_descriptor_selector_fixed_on_train():
    D = compute_descriptors(SMILES)
    sel = DescriptorSelector().fit(D.iloc[:8])
    assert sel.transform(D.iloc[8:]).shape[1] == len(sel.columns)
    assert sel.summary()["n_dropped_constant"] > 0  # 학습 8개에서 상수인 컬럼이 있다


def test_sample_params_within_space():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = sample_params(LGBM_SPACE, rng)
        for k, (_, lo, hi) in LGBM_SPACE.items():
            assert lo <= p[k] <= hi


def test_random_search_reproducible(toy):
    X = ecfp4(toy.smiles).astype(np.float32)
    y = toy.label.to_numpy()

    def fn(params, seed):
        return make_lgbm("regression", params, seed)

    a = random_search(fn, LGBM_SPACE, X[:120], y[:120], X[120:], y[120:], n_trials=3, seed=5, task="regression")
    b = random_search(fn, LGBM_SPACE, X[:120], y[:120], X[120:], y[120:], n_trials=3, seed=5, task="regression")
    assert a == b


def test_run_grid_writes_and_resumes(toy, tmp_path):
    out = tmp_path / "r.csv"
    kw = dict(dataset="toy", task="regression", target_task="y", features=FeatureStore.build(toy),
              models=["A_ecfp_lgbm", "B_desc_lgbm"], strategies=["random"], split_seeds=[0, 1],
              budgets=[50, None], n_trials=2, out_path=out, log=lambda _: None)
    res = run_grid(toy, **kw)
    n_combos = 2 * 2 * 2
    assert res.groupby(["model", "split_seed", "n_labels"]).ngroups == n_combos
    assert (res.leakage_exact == 0).all() and (res.hp_n_trials == 2).all()
    assert set(res.n_train[res.n_labels == "50"]) == {50}
    before = len(res)
    assert len(run_grid(toy, **kw)) == before  # 재실행 시 건너뛴다


def test_run_grid_proportional_valid(toy, tmp_path):
    res = run_grid(toy, dataset="toy", task="regression", target_task="y", features=FeatureStore.build(toy),
                   models=["A_ecfp_lgbm"], strategies=["random"], split_seeds=[0], budgets=[40], n_trials=1,
                   out_path=tmp_path / "p.csv", valid_policy="proportional", log=lambda _: None)
    assert set(res.n_valid) == {5}  # 40 × 0.1/0.8


def test_checkpoint_equals_shorter_search(toy):
    """n_trials=6 탐색의 앞 3회 최고값 == n_trials=3 탐색 결과 (체크포인트 방식의 전제)."""
    from labelbudget.models.gbm import best_of, search_trace
    X = ecfp4(toy.smiles).astype(np.float32)
    y = toy.label.to_numpy()

    def fn(params, seed):
        return make_lgbm("regression", params, seed)

    trace = search_trace(fn, LGBM_SPACE, X[:120], y[:120], X[120:], y[120:], n_trials=6, seed=3, task="regression")
    short = random_search(fn, LGBM_SPACE, X[:120], y[:120], X[120:], y[120:], n_trials=3, seed=3, task="regression")
    assert best_of(trace, 3, "regression") == short
    scores = [best_of(trace, k, "regression")[1] for k in range(1, 7)]
    assert scores == sorted(scores, reverse=True)  # 시도가 늘수록 valid RMSE는 줄거나 같다


def test_run_grid_trial_checkpoints(toy, tmp_path):
    res = run_grid(toy, dataset="toy", task="regression", target_task="y", features=FeatureStore.build(toy),
                   models=["B_desc_lgbm"], strategies=["random"], split_seeds=[0], budgets=[None], n_trials=4,
                   trial_checkpoints=[1, 2, 99], out_path=tmp_path / "c.csv", log=lambda _: None)
    assert sorted(res.hp_n_trials.unique()) == [1, 2, 4]  # 99는 n_trials 초과라 제외, n_trials는 항상 포함
    rmse = res[res.metric_name == "rmse"].set_index("hp_n_trials")
    assert rmse.hp_valid_score[4] <= rmse.hp_valid_score[1]
