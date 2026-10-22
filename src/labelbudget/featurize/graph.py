"""분자 → 그래프 텐서. chemprop v2 기본 featurizer(`V2` 원자, `MultiHot` 결합)와 같은 정의. 스펙 04 §2.1.

재현 비교를 위해 chemprop 2.3.1 소스의 정의를 그대로 옮겼다 (`test_mpnn_equivalence.py`에서 대조).
결합은 방향마다 하나씩, 2k와 2k+1이 서로의 역방향이다.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType, HybridizationType

ATOMIC_NUMS = list(range(1, 37)) + [53]
DEGREES = list(range(6))
FORMAL_CHARGES = [-1, -2, 1, 2, 0]
CHIRAL_TAGS = list(range(4))
NUM_HS = list(range(5))
HYBRIDIZATIONS = [HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP2D,
                  HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2]
BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
BOND_STEREOS = list(range(6))

_ATOM_CHOICES = [ATOMIC_NUMS, DEGREES, FORMAL_CHARGES, CHIRAL_TAGS, NUM_HS, HYBRIDIZATIONS]
ATOM_FDIM = sum(len(c) + 1 for c in _ATOM_CHOICES) + 2       # 미지값 칸 포함 + 방향족 + 질량 = 72
BOND_FDIM = 1 + len(BOND_TYPES) + 2 + len(BOND_STEREOS) + 1   # null + 종류 + 공액·고리 + 입체 = 14


def atom_features(atom: Chem.Atom) -> np.ndarray:
    x = np.zeros(ATOM_FDIM, dtype=np.float32)
    values = [atom.GetAtomicNum(), atom.GetTotalDegree(), atom.GetFormalCharge(), int(atom.GetChiralTag()),
              int(atom.GetTotalNumHs()), atom.GetHybridization()]
    i = 0
    for v, choices in zip(values, _ATOM_CHOICES):
        x[i + (choices.index(v) if v in choices else len(choices))] = 1
        i += len(choices) + 1
    x[i] = int(atom.GetIsAromatic())
    x[i + 1] = 0.01 * atom.GetMass()
    return x


def bond_features(bond: Chem.Bond) -> np.ndarray:
    # chemprop 2.3.1과 같게: 목록에 없는 결합 종류의 칸은 '공액' 칸과 겹쳐 공액 값으로 덮어써진다 (원본 동작 보존)
    x = np.zeros(BOND_FDIM, dtype=np.float32)
    bt = bond.GetBondType()
    x[1 + (BOND_TYPES.index(bt) if bt in BOND_TYPES else len(BOND_TYPES))] = 1
    i = 1 + len(BOND_TYPES)
    x[i] = int(bond.GetIsConjugated())
    x[i + 1] = int(bond.IsInRing())
    i += 2
    st = int(bond.GetStereo())
    x[i + (BOND_STEREOS.index(st) if st in BOND_STEREOS else len(BOND_STEREOS))] = 1
    return x


@dataclass(frozen=True)
class MolGraph:
    V: np.ndarray            # (n_atoms, ATOM_FDIM)
    E: np.ndarray            # (2·n_bonds, BOND_FDIM)
    edge_index: np.ndarray   # (2, 2·n_bonds) — [출발 원자, 도착 원자]
    rev: np.ndarray          # (2·n_bonds,) 역방향 결합 인덱스


def mol_graph(smiles: str) -> MolGraph:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES 파싱 실패: {smiles}")
    V = np.stack([atom_features(a) for a in mol.GetAtoms()]) if mol.GetNumAtoms() else np.zeros((1, ATOM_FDIM),
                                                                                                   np.float32)
    n_bonds = mol.GetNumBonds()
    E = np.zeros((2 * n_bonds, BOND_FDIM), dtype=np.float32)
    edge_index = np.zeros((2, 2 * n_bonds), dtype=np.int64)
    for k, b in enumerate(mol.GetBonds()):
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        E[2 * k] = E[2 * k + 1] = bond_features(b)
        edge_index[:, 2 * k] = (u, v)
        edge_index[:, 2 * k + 1] = (v, u)
    rev = np.arange(2 * n_bonds).reshape(-1, 2)[:, ::-1].ravel().copy()
    return MolGraph(V, E, edge_index, rev)


@dataclass
class BatchGraph:
    V: torch.Tensor
    E: torch.Tensor
    edge_index: torch.Tensor
    rev: torch.Tensor
    batch: torch.Tensor      # 원자 → 분자 번호
    n_mols: int

    def to(self, device=None, dtype=None) -> BatchGraph:
        f = dict(device=device, dtype=dtype)
        return BatchGraph(self.V.to(**f), self.E.to(**f), self.edge_index.to(device), self.rev.to(device),
                          self.batch.to(device), self.n_mols)


def batch_graphs(graphs: Sequence[MolGraph]) -> BatchGraph:
    Vs, Es, eis, revs, bs = [], [], [], [], []
    atom_off = edge_off = 0
    for i, g in enumerate(graphs):
        Vs.append(g.V)
        Es.append(g.E)
        eis.append(g.edge_index + atom_off)
        revs.append(g.rev + edge_off)
        bs.append(np.full(len(g.V), i, dtype=np.int64))
        atom_off += len(g.V)
        edge_off += len(g.E)
    return BatchGraph(
        V=torch.from_numpy(np.concatenate(Vs)),
        E=torch.from_numpy(np.concatenate(Es)) if edge_off else torch.zeros((0, BOND_FDIM)),
        edge_index=torch.from_numpy(np.concatenate(eis, axis=1)),
        rev=torch.from_numpy(np.concatenate(revs)),
        batch=torch.from_numpy(np.concatenate(bs)),
        n_mols=len(graphs),
    )
