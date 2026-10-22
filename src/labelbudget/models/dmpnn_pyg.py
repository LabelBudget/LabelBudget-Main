"""D-MPNN의 PyG `MessagePassing` 버전. 등가성 비교 전용 (스펙 04 §2.3, 결정 G2).

PyG에는 D-MPNN 레이어가 없으므로, 결합→원자 합산만 PyG `propagate`(aggr="add")에 맡기고
나머지 계산은 직접 구현(`dmpnn.py`)과 같게 둔다. 두 구현이 같은 가중치로 같은 값을 내는지 확인한다.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing

from labelbudget.featurize.graph import BatchGraph
from labelbudget.models.dmpnn import DMPNNEncoder


class _EdgeToAtomSum(MessagePassing):
    """방향 결합 상태를 도착 원자로 합산한다."""

    def __init__(self):
        super().__init__(aggr="add", flow="source_to_target")

    def forward(self, h_edge: Tensor, edge_index: Tensor, n_atoms: int) -> Tensor:
        return self.propagate(edge_index, h_edge=h_edge, size=(n_atoms, n_atoms))

    def message(self, h_edge: Tensor) -> Tensor:
        return h_edge


class DMPNNEncoderPyG(nn.Module):
    """`DMPNNEncoder`와 같은 파라미터 이름·모양 (state_dict 그대로 공유 가능)."""

    def __init__(self, ref: DMPNNEncoder):
        super().__init__()
        self.W_i, self.W_h, self.W_o = ref.W_i, ref.W_h, ref.W_o
        self.depth, self.dropout, self.d_h = ref.depth, ref.dropout, ref.d_h
        self.agg = _EdgeToAtomSum()

    def forward(self, g: BatchGraph) -> Tensor:
        src = g.edge_index[0]
        n_atoms = g.V.shape[0]
        z = self.W_i(torch.cat([g.V[src], g.E], dim=1))
        h = torch.relu(z)
        for _ in range(1, self.depth):
            m = self.agg(h, g.edge_index, n_atoms)[src] - h[g.rev]
            h = self.dropout(torch.relu(z + self.W_h(m)))
        m_v = self.agg(h, g.edge_index, n_atoms) if h.shape[0] else g.V.new_zeros((n_atoms, self.d_h))
        return self.dropout(torch.relu(self.W_o(torch.cat([g.V, m_v], dim=1))))
