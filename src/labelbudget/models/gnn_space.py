"""D-MPNN 탐색 공간. torch 없이 import되도록 gnn.py에서 분리했다.

LightGBM(conda libomp)과 torch(자체 libomp)가 한 프로세스에 함께 로드되면
LightGBM 멀티스레드 학습이 세그폴트로 죽는다. 그래서 베이스라인 경로는 torch를 import하지 않는다.
"""
from __future__ import annotations

# 탐색 공간 (초안, 스펙 04 §2.4). 종류는 gbm.sample_params와 같은 규약
GNN_SPACE: dict[str, tuple] = {
    "d_h": ("int_log", 128, 512),
    "depth": ("int", 2, 6),
    "dropout": ("float", 0.0, 0.4),
    "lr": ("log", 1e-4, 3e-3),
    "ffn_hidden": ("int_log", 128, 512),
    "ffn_layers": ("int", 1, 3),
    "batch_size": ("choice", 32, 64),
}
# C_dmpnn_v2 (04 §6.3, 2026-10-25): readout을 탐색한다. 원인 진단(§6.2)에서 mean readout이 대예산 격차의 원인이었다.
# GNN_SPACE는 바꾸지 않는다 (기존 C_dmpnn 결과 재현용)
GNN_SPACE_V2: dict[str, tuple] = {**GNN_SPACE, "readout": ("choice", "mean", "norm")}
