import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from labelbudget.featurize.graph import batch_graphs, mol_graph
from labelbudget.models.dmpnn import DMPNNEncoder
from labelbudget.models.gnn_space import FIXED_BACKBONE
from labelbudget.pretrain.tasks import atom_classes, drop_atoms, graphcl_loss, mask_atoms

ROOT = Path(__file__).resolve().parents[1]


def test_mask_hides_atom_from_both_paths():
    # 마지막 원자만 다른 두 분자 (C-C-O vs C-C-N). 결합 피처와 나머지 원자 피처는 같다.
    # 그 원자를 가리면 인코더 출력이 같아야 한다 → W_i(결합 메시지)와 W_o(원자 출력) 모두에서 가려졌다는 뜻
    a, b = mol_graph("CCO"), mol_graph("CCN")
    enc = DMPNNEncoder(d_h=16, depth=3).eval()
    with torch.no_grad():
        assert not torch.equal(enc(batch_graphs([a])), enc(batch_graphs([b])))  # 대조: 가리기 전에는 다르다
        a.V[2], b.V[2] = 0.0, 0.0
        assert np.array_equal(a.V, b.V) and np.array_equal(a.E, b.E)
        assert torch.equal(enc(batch_graphs([a])), enc(batch_graphs([b])))


def test_mask_atoms_counts_and_targets():
    g = mol_graph("CC(=O)Nc1ccc(O)cc1")
    rng = np.random.default_rng(0)
    mg, idx = mask_atoms(g, rng)
    assert len(idx) == int(len(g.V) * 0.15 + 1)
    assert not mg.V[idx].any() and mg.V[np.setdiff1d(np.arange(len(g.V)), idx)].any(axis=1).all()
    assert g.V[idx].any(axis=1).all()  # 원본은 그대로
    cls = atom_classes(g)
    assert cls.max() < 38 and set(cls) == {5, 6, 7}  # C, N, O (원자 번호 − 1)
    mg2, idx2 = mask_atoms(g, rng, rate=0.2, count="graphcl", fill="mean")
    assert len(idx2) == int(len(g.V) * 0.2)
    assert np.allclose(mg2.V[idx2], g.V.mean(axis=0))  # GraphCL maskN: 분자 평균 피처


def test_drop_atoms_keeps_graph_consistent():
    rng = np.random.default_rng(1)
    for s in ["CC(=O)Nc1ccc(O)cc1", "c1ccc2ccccc2c1", "CCO", "C"]:
        g = mol_graph(s)
        for _ in range(20):
            d = drop_atoms(g, rng)
            n = len(g.V)
            assert len(d.V) == n - min(int(n * 0.2), n - 1)
            if len(d.E):
                assert d.edge_index.max() < len(d.V)
                assert np.array_equal(d.edge_index[:, d.rev], d.edge_index[::-1])  # rev가 역방향을 가리킨다
                assert np.array_equal(d.E[0::2], d.E[1::2])
            batch_graphs([d, g])  # 배치로 묶여야 한다


def test_graphcl_loss_matches_formula():
    torch.manual_seed(0)
    x1, x2 = torch.randn(5, 8), torch.randn(5, 8)
    s = torch.exp(torch.nn.functional.cosine_similarity(x1[:, None], x2[None], dim=2) / 0.1)
    manual = -torch.log(s.diagonal() / (s.sum(1) - s.diagonal())).mean()
    assert torch.allclose(graphcl_loss(x1, x2), manual, atol=1e-5)
    assert graphcl_loss(x1, x1 + 0.01 * x2) < graphcl_loss(x1, x2)


def test_pretrain_script_smoke(tmp_path):
    smi = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCOC(=O)C", "c1ccncc1", "CC(C)O", "OCCO"] * 4
    data = tmp_path / "z.csv"
    pd.DataFrame({"smiles": smi, "inchikey": [f"K{i}" for i in range(len(smi))]}).to_csv(data, index=False)
    for method in ("mask", "contrast"):
        cmd = [sys.executable, str(ROOT / "scripts" / "pretrain.py"), "--method", method, "--device", "cpu",
               "--epochs", "2", "--batch-size", "8", "--data", str(data), "--out-dir", str(tmp_path / "w")]
        subprocess.run(cmd, check=True, capture_output=True, cwd=ROOT)
        meta = json.loads((tmp_path / "w" / f"D_{method}.json").read_text())
        assert len(meta["history"]) == 2 and np.isfinite(meta["history"][-1]["train_loss"])
        enc = DMPNNEncoder(**FIXED_BACKBONE)
        enc.load_state_dict(torch.load(tmp_path / "w" / f"D_{method}.pt"))  # 미세조정 인코더에 그대로 들어간다
        # 같은 명령을 다시 돌리면 체크포인트에서 이어서(이미 끝났으므로 에폭 추가 없이) 끝난다
        subprocess.run(cmd, check=True, capture_output=True, cwd=ROOT)
        assert len(json.loads((tmp_path / "w" / f"D_{method}.json").read_text())["history"]) == 2
