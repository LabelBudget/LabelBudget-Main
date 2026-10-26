"""D-MPNN 직접 구현의 수치 등가성. 스펙 04 §2.3.

- 직접 구현 vs PyG 버전: 본 환경 (`labelbudget`)
- 직접 구현 vs chemprop 2.3.1 공식 구현, 피처 정의 대조: chemprop 전용 환경 (`labelbudget-equiv`)
  chemprop은 RDKit을 2026.03으로 내려 본 환경과 분리했다. 해당 환경에서만 실행되고 그 외에는 skip.
모두 CPU float64, 기준 max|Δ| < 1e-5.
"""
import numpy as np
import pytest
import torch

from labelbudget.featurize.graph import ATOM_FDIM, BOND_FDIM, batch_graphs, mol_graph
from labelbudget.models.dmpnn import DMPNN, DMPNNEncoder, readout

TOL = 1e-5
SMILES = [
    "CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "C[C@H](N)C(=O)O", "C/C=C/C(=O)O", "O=C1NC(=O)C(=O)N1",
    "CN1CCC[C@H]1c1cccnc1", "FC(F)(F)c1ccc(Cl)cc1", "CCOP(=S)(OCC)Oc1ccc([N+](=O)[O-])cc1", "C#CCO",
    "c1ccc2c(c1)[nH]c1ccccc12", "OCC(O)C1OC(=O)C(O)=C1O", "CC(C)(C)c1ccc(O)cc1", "Brc1ccc(I)cc1",
    "C1CC2CCC1C2", "O=S(=O)(O)c1ccccc1", "CC(=O)OC1CCCCC1", "N#Cc1ccccc1", "C=CC(=O)OC", "CSC",
]


def _encoder(depth=3, d_h=32, seed=0):
    torch.manual_seed(seed)
    return DMPNNEncoder(d_h=d_h, depth=depth).double().eval()


def _batch(smiles=SMILES):
    return batch_graphs([mol_graph(s) for s in smiles]).to(dtype=torch.float64)


def test_feature_dims():
    g = mol_graph("CCO")
    assert g.V.shape == (3, ATOM_FDIM) == (3, 72) and g.E.shape == (4, BOND_FDIM) == (4, 14)
    assert (g.edge_index[:, g.rev] == g.edge_index[::-1]).all()  # 역방향 인덱스


def test_reverse_message_excluded_two_atoms():
    """2원자 분자: 각 결합으로 들어오는 메시지는 자기 역방향뿐이라 m=0 → 깊이와 무관하게 h가 초기값과 같다."""
    g = _batch(["CO"])
    shallow, deep = _encoder(depth=1), _encoder(depth=5)
    deep.load_state_dict(shallow.state_dict())
    with torch.no_grad():
        assert torch.allclose(shallow(g), deep(g), atol=1e-12)


def test_batch_independence():
    enc = _encoder()
    with torch.no_grad():
        together = enc(_batch())
        separate = torch.cat([enc(_batch([s])) for s in SMILES])
    assert (together - separate).abs().max().item() < TOL


def test_readout_mean_matches_manual():
    g = _batch(["CCO", "c1ccccc1"])
    h = torch.arange(g.V.shape[0] * 2, dtype=torch.float64).reshape(-1, 2)
    out = readout(h, g.batch, g.n_mols, "mean")
    assert torch.allclose(out[0], h[:3].mean(0)) and torch.allclose(out[1], h[3:].mean(0))


def test_readout_norm_is_sum_over_100():
    g = _batch(["CCO", "c1ccccc1"])
    h = torch.arange(g.V.shape[0] * 2, dtype=torch.float64).reshape(-1, 2)
    out = readout(h, g.batch, g.n_mols, "norm")
    assert torch.allclose(out[0], h[:3].sum(0) / 100) and torch.allclose(out[1], h[3:].sum(0) / 100)


def test_dmpnn_forward_shape():
    model = DMPNN(d_h=16, depth=2, ffn_hidden=8).double()
    assert model(_batch()).shape == (len(SMILES),)


def test_direct_vs_pyg():
    pytest.importorskip("torch_geometric")
    from labelbudget.models.dmpnn_pyg import DMPNNEncoderPyG
    for depth in (1, 2, 3, 5):
        enc = _encoder(depth=depth)
        pyg = DMPNNEncoderPyG(enc).double().eval()
        with torch.no_grad():
            diff = (enc(_batch()) - pyg(_batch())).abs().max().item()
        assert diff < TOL, f"depth={depth}: {diff}"


# ── chemprop 전용 환경 ─────────────────────────────────────────────

def _chemprop():
    return pytest.importorskip("chemprop")


def test_featurizer_matches_chemprop():
    _chemprop()
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem
    feat = SimpleMoleculeMolGraphFeaturizer()
    for s in SMILES:
        ref, ours = feat(Chem.MolFromSmiles(s)), mol_graph(s)
        assert np.allclose(ref.V, ours.V) and np.allclose(ref.E, ours.E), s
        assert (np.asarray(ref.edge_index) == ours.edge_index).all() and (np.asarray(ref.rev_edge_index) == ours.rev).all()


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_direct_vs_chemprop(depth):
    _chemprop()
    from chemprop.data import BatchMolGraph
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.nn import BondMessagePassing
    from rdkit import Chem

    enc = _encoder(depth=depth)
    ref = BondMessagePassing(d_v=ATOM_FDIM, d_e=BOND_FDIM, d_h=enc.d_h, depth=depth).double().eval()
    ref.load_state_dict({k: v for k, v in enc.state_dict().items()}, strict=True)

    feat = SimpleMoleculeMolGraphFeaturizer()
    bmg = BatchMolGraph([feat(Chem.MolFromSmiles(s)) for s in SMILES])
    bmg.V, bmg.E = bmg.V.double(), bmg.E.double()
    with torch.no_grad():
        diff = (enc(_batch()) - ref(bmg)).abs().max().item()
    assert diff < TOL, f"depth={depth}: {diff}"


@pytest.mark.parametrize("how,agg", [("mean", "MeanAggregation"), ("norm", "NormAggregation")])
def test_readout_vs_chemprop(how, agg):
    _chemprop()
    import chemprop.nn as cnn

    g = _batch()
    h = torch.randn(g.V.shape[0], 8, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    ref = getattr(cnn, agg)()(h, g.batch)
    assert (readout(h, g.batch, g.n_mols, how) - ref).abs().max().item() < TOL
