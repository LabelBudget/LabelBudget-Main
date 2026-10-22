"""D-MPNN 직접 구현 (torch `index_add_`만 사용, PyG·torch_scatter 상속/의존 없음). 스펙 04 §2.2.

    z_{v→w}       = W_i · [x_v ‖ e_vw]                       (활성화 전 값 보관)
    h¹            = ReLU(z)
    m_{v→w}       = Σ_{k→v} h_{k→v} − h_{w→v}               (역방향 메시지 제외)
    h^{t+1}       = Dropout(ReLU(z + W_h · m))              (depth−1회)
    m_v           = Σ_{k→v} h_{k→v}
    h_v           = Dropout(ReLU(W_o · [x_v ‖ m_v]))

chemprop 2.3.1 `BondMessagePassing`과 같은 계산 (`tests/test_mpnn_equivalence.py`).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from labelbudget.featurize.graph import ATOM_FDIM, BOND_FDIM, BatchGraph


def scatter_sum(src: Tensor, index: Tensor, n: int) -> Tensor:
    """src의 행을 index 위치로 합친다 (n행 출력)."""
    return src.new_zeros((n, src.shape[1])).index_add_(0, index, src)


class DMPNNEncoder(nn.Module):
    def __init__(self, d_v: int = ATOM_FDIM, d_e: int = BOND_FDIM, d_h: int = 300, depth: int = 3,
                 dropout: float = 0.0, bias: bool = False):
        super().__init__()
        if depth < 1:
            raise ValueError("depth는 1 이상")
        self.W_i = nn.Linear(d_v + d_e, d_h, bias=bias)
        self.W_h = nn.Linear(d_h, d_h, bias=bias)
        self.W_o = nn.Linear(d_v + d_h, d_h)
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.d_h = d_h

    def forward(self, g: BatchGraph) -> Tensor:
        """원자 표현 (n_atoms, d_h)."""
        src, dst = g.edge_index
        n_atoms = g.V.shape[0]
        z = self.W_i(torch.cat([g.V[src], g.E], dim=1))
        h = torch.relu(z)
        for _ in range(1, self.depth):
            m = scatter_sum(h, dst, n_atoms)[src] - h[g.rev]
            h = self.dropout(torch.relu(z + self.W_h(m)))
        m_v = scatter_sum(h, dst, n_atoms) if h.shape[0] else g.V.new_zeros((n_atoms, self.d_h))
        return self.dropout(torch.relu(self.W_o(torch.cat([g.V, m_v], dim=1))))


def readout(h_atoms: Tensor, batch: Tensor, n_mols: int, how: str = "mean") -> Tensor:
    """원자 표현 → 분자 표현. mean | sum | norm(합/100, chemprop 기본)."""
    total = scatter_sum(h_atoms, batch, n_mols)
    if how == "sum":
        return total
    if how == "norm":
        return total / 100.0
    if how == "mean":
        counts = torch.bincount(batch, minlength=n_mols).clamp(min=1).to(total.dtype)
        return total / counts[:, None]
    raise ValueError(f"알 수 없는 readout '{how}'")


class DMPNN(nn.Module):
    """인코더 + readout + FFN. 회귀는 실수 1개, 분류는 로짓 1개를 출력한다."""

    def __init__(self, d_h: int = 300, depth: int = 3, dropout: float = 0.0, ffn_hidden: int = 300,
                 ffn_layers: int = 1, readout: str = "mean"):
        super().__init__()
        self.encoder = DMPNNEncoder(d_h=d_h, depth=depth, dropout=dropout)
        self.readout_how = readout
        layers: list[nn.Module] = []
        d_in = d_h
        for _ in range(ffn_layers):
            layers += [nn.Dropout(dropout), nn.Linear(d_in, ffn_hidden), nn.ReLU()]
            d_in = ffn_hidden
        layers += [nn.Dropout(dropout), nn.Linear(d_in, 1)]
        self.ffn = nn.Sequential(*layers)

    def embed(self, g: BatchGraph) -> Tensor:
        return readout(self.encoder(g), g.batch, g.n_mols, self.readout_how)

    def forward(self, g: BatchGraph) -> Tensor:
        return self.ffn(self.embed(g)).squeeze(-1)
