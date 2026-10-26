import numpy as np
import pandas as pd
import pytest
import torch

from labelbudget.evaluation.runner import FeatureStore, run_grid
from labelbudget.featurize.graph import mol_graph
from labelbudget.models.gbm import LGBM_SPACE, sample_params
from labelbudget.models.gnn import GNN_SPACE, DMPNNEstimator
from labelbudget.models.gnn_space import GNN_SPACE_V2

SMILES = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "c1ccncc1", "Cc1ccncc1",
          "C1CCCCC1", "CC1CCCCC1", "CCOC(=O)C", "CC(=O)O", "CCN", "CCCN", "c1ccc2ccccc2c1", "Oc1ccccc1"]
FAST = {"d_h": 32, "depth": 2, "dropout": 0.0, "lr": 3e-3, "ffn_hidden": 32, "ffn_layers": 1, "batch_size": 16}


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    smiles = [SMILES[i % len(SMILES)] + "C" * (i // len(SMILES)) for i in range(128)]
    y = np.array([s.count("C") * 0.3 + s.count("c") * 0.1 + rng.normal(scale=0.05) for s in smiles])
    return smiles, y


def test_sample_params_choice_and_lgbm_stream_unchanged():
    p = sample_params(GNN_SPACE, np.random.default_rng(0))
    assert p["batch_size"] in (32, 64) and 2 <= p["depth"] <= 6
    # choice 지원 추가가 LightGBM 후보열을 바꾸지 않았는지 (진행 중 그리드의 resume 일관성)
    a = sample_params(LGBM_SPACE, np.random.default_rng(1004))
    assert a == sample_params(LGBM_SPACE, np.random.default_rng(1004))
    assert set(a) == set(LGBM_SPACE)


def test_space_v2_adds_readout_only():
    assert {k: v for k, v in GNN_SPACE_V2.items() if k != "readout"} == GNN_SPACE
    assert GNN_SPACE_V2["readout"] == ("choice", "mean", "norm")
    assert {sample_params(GNN_SPACE_V2, np.random.default_rng(s))["readout"] for s in range(20)} == {"mean", "norm"}


def test_estimator_readout_param(toy):
    smiles, y = toy
    G = [mol_graph(s) for s in smiles]
    kw = dict(valid=(G[96:], y[96:]), max_epochs=1)
    assert DMPNNEstimator("regression", FAST, seed=0, **kw).fit(G[:96], y[:96]).model.readout_how == "mean"
    est = DMPNNEstimator("regression", {**FAST, "readout": "norm"}, seed=0, **kw).fit(G[:96], y[:96])
    assert est.model.readout_how == "norm"


def test_estimator_learns_and_is_deterministic(toy):
    smiles, y = toy
    perm = np.random.default_rng(1).permutation(len(smiles))  # 무작위 분할 (앞뒤로 자르면 외삽이 된다)
    G = [mol_graph(smiles[i]) for i in perm]
    y = y[perm]
    Gtr, Gva, ytr, yva = G[:96], G[96:], y[:96], y[96:]

    def fit():
        return DMPNNEstimator("regression", FAST, seed=0, valid=(Gva, yva), max_epochs=30, patience=30).fit(Gtr, ytr)

    a, b = fit(), fit()
    rmse = np.sqrt(((a.predict(Gva) - yva) ** 2).mean())
    assert rmse < yva.std()                        # 평균 예측보다 낫다
    assert np.array_equal(a.predict(Gva), b.predict(Gva))  # 같은 시드 → 비트 단위로 같은 결과 (CPU 결정론 모드)
    assert a.history[a.best_epoch] == max(a.history)  # 최고 valid 시점의 가중치를 복원


def test_estimator_classification_outputs(toy):
    smiles, y = toy
    G = [mol_graph(s) for s in smiles]
    yc = (y > np.median(y)).astype(int)
    est = DMPNNEstimator("classification", FAST, seed=0, valid=(G[96:], yc[96:]), max_epochs=5).fit(G[:96], yc[:96])
    p = est.predict_proba(G[96:])
    assert p.shape == (32, 2) and np.allclose(p.sum(1), 1) and ((0 <= p) & (p <= 1)).all()


def test_run_grid_with_gnn(toy, tmp_path):
    smiles, y = toy
    df = pd.DataFrame({"smiles": smiles, "inchikey": [f"K{i:04d}-X-N" for i in range(len(smiles))], "label": y})
    torch.set_num_threads(1)
    res = run_grid(df, dataset="toy", task="regression", target_task="y", features=FeatureStore.build(df),
                   models=["C_dmpnn", "B_desc_lgbm"], strategies=["random"], split_seeds=[0], budgets=[None],
                   n_trials=2, trial_checkpoints=[1], out_path=tmp_path / "g.csv",
                   gnn_opts={"max_epochs": 3, "patience": 3}, log=lambda _: None)
    assert set(res.model) == {"C_dmpnn", "B_desc_lgbm"} and sorted(res.hp_n_trials.unique()) == [1, 2]
    assert res.groupby("model").config_hash.nunique().eq(1).all()
    assert res[res.model == "C_dmpnn"].config_hash.iloc[0] != res[res.model == "B_desc_lgbm"].config_hash.iloc[0]


def test_deterministic_mode_is_scoped_and_cpu_only(toy):
    smiles, y = toy
    G = [mol_graph(s) for s in smiles]
    before = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    DMPNNEstimator("regression", FAST, seed=0, valid=(G[96:], y[96:]), max_epochs=2, threads=1).fit(G[:96], y[:96])
    assert (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()) == before  # 전역 설정을 남기지 않는다
    with pytest.raises(ValueError, match="CPU"):
        DMPNNEstimator("regression", FAST, seed=0, valid=(G, y), device="mps")
