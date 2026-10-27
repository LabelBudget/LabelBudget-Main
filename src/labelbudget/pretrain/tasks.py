"""자기지도 사전학습 과제 2종. 스펙 05 §2 P3.

- 마스킹 (Hu et al. 2020 AttrMasking): 분자마다 int(n·0.15 + 1)개 원자의 피처를 0으로 바꾸고 원자 종류 38칸을 맞힌다
- 대조학습 (GraphCL, You et al. 2020): 뷰 1 = 원자 피처 마스킹, 뷰 2 = 원자 삭제 (비율 각 0.2, 개수 int(n·0.2)).
  평균 풀링 → projection(300→300→300) → 손실. 온도·비율·projection·풀링·손실 형태는 GraphCL 공개 코드
  (transferLearning_MoleculeNet_PPI/chem/pretrain_graphcl.py, loader.py)의 값을 그대로 쓴다.
  maskN은 GraphCL 화학 코드처럼 가린 원자 피처를 그 분자의 평균 원자 피처로 바꾼다 (마스킹 과제는 0).
  GraphCL 화학 실험의 증강 조합(dropN·subgraph 무작위)과 달리 뷰 2개를 maskN·dropN으로 고정한다 (05 §2 P3)

D-MPNN의 첫 메시지 W_i·[x_v ‖ e_vw]와 W_o·[x_v ‖ m_v]가 모두 g.V를 읽으므로, MolGraph의 V 행을 0으로 바꾸면
두 경로 모두에서 원자 정보가 가려진다 (tests/test_pretrain.py).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from labelbudget.featurize.graph import ATOMIC_NUMS, MolGraph

N_ATOM_CLASSES = len(ATOMIC_NUMS) + 1  # 38: 원자 번호 원-핫의 첫 칸들 + 미지
MASK_RATE = 0.15
AUG_RATIO = 0.2
TEMPERATURE = 0.1


def atom_classes(g: MolGraph) -> np.ndarray:
    return g.V[:, :N_ATOM_CLASSES].argmax(1)


def mask_atoms(g: MolGraph, rng: np.random.Generator, rate: float = MASK_RATE,
               count: str = "hu", fill: str = "zero") -> tuple[MolGraph, np.ndarray]:
    """원자 피처를 가린 그래프와 가린 원자 위치.

    count='hu'는 int(n·rate+1)개 (Hu et al.), 'graphcl'은 int(n·rate)개. fill='zero'는 0, 'mean'은 분자 평균 피처 (GraphCL)
    """
    n = len(g.V)
    k = min(n, int(n * rate + 1)) if count == "hu" else int(n * rate)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    V = g.V.copy()
    V[idx] = 0.0 if fill == "zero" else g.V.mean(axis=0)
    return MolGraph(V, g.E, g.edge_index, g.rev), idx


def drop_atoms(g: MolGraph, rng: np.random.Generator, ratio: float = AUG_RATIO) -> MolGraph:
    """int(n·ratio)개 원자와 그 원자에 닿는 결합(양방향)을 지운다. 결합 2k·2k+1 짝 구조와 rev를 다시 만든다."""
    n = len(g.V)
    k = min(int(n * ratio), n - 1)
    if k <= 0:
        return g
    keep = np.ones(n, dtype=bool)
    keep[rng.choice(n, size=k, replace=False)] = False
    new_id = np.cumsum(keep) - 1
    u, v = g.edge_index[0, 0::2], g.edge_index[1, 0::2]  # 결합 k의 양 끝 (열 2k가 u→v)
    pairs = np.flatnonzero(keep[u] & keep[v])
    e_idx = np.stack([2 * pairs, 2 * pairs + 1], axis=1).ravel()
    edge_index = new_id[g.edge_index[:, e_idx]]
    rev = np.arange(len(e_idx)).reshape(-1, 2)[:, ::-1].ravel().copy()
    return MolGraph(g.V[keep], g.E[e_idx], edge_index.astype(np.int64), rev)


class MaskHead(nn.Module):
    """원자 표현 → 원자 종류 로짓 (Hu et al.의 선형 헤드)."""

    def __init__(self, d_h: int = 300):
        super().__init__()
        self.lin = nn.Linear(d_h, N_ATOM_CLASSES)

    def forward(self, h_atoms: Tensor) -> Tensor:
        return self.lin(h_atoms)


class ProjectionHead(nn.Module):
    """GraphCL projection_head: Linear(300,300) → ReLU → Linear(300,300)."""

    def __init__(self, d_h: int = 300):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_h, d_h), nn.ReLU(inplace=True), nn.Linear(d_h, d_h))

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


def graphcl_loss(x1: Tensor, x2: Tensor, temperature: float = TEMPERATURE) -> Tensor:
    """GraphCL loss_cl 그대로: 뷰1의 i와 뷰2의 i가 양성, 뷰2의 나머지가 음성 (같은 뷰 음성·대칭항 없음)."""
    sim = torch.einsum("ik,jk->ij", x1, x2) / torch.einsum("i,j->ij", x1.norm(dim=1), x2.norm(dim=1))
    sim = torch.exp(sim / temperature)
    pos = sim.diagonal()
    return -torch.log(pos / (sim.sum(dim=1) - pos)).mean()
