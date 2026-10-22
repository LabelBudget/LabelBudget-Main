"""D-MPNN 학습 래퍼 (sklearn 형태). 비교군 C. 스펙 04 §2.4.

`fit(X, y)`의 X는 MolGraph 목록. valid는 조기 종료용으로 생성 시 받는다 —
LightGBM과 같은 `search_trace`/체크포인트 경로를 그대로 쓰기 위해서다.
"""
from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from labelbudget.evaluation.metrics import classification_metrics, regression_metrics
from labelbudget.featurize.graph import MolGraph, batch_graphs
from labelbudget.models.dmpnn import DMPNN
from labelbudget.models.gnn_space import GNN_SPACE  # noqa: F401  (기존 import 경로 유지)


@contextmanager
def _deterministic(on: bool) -> Iterator[None]:
    """멀티스레드 index_add_의 누적 순서를 고정한다. 전역 설정이라 끝나면 이전 값으로 되돌린다."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(on or prev)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


class DMPNNEstimator:
    def __init__(self, task: str, params: dict, seed: int, *, valid: tuple[Sequence[MolGraph], np.ndarray],
                 max_epochs: int = 100, patience: int = 10, device: str = "cpu", deterministic: bool = True,
                 eval_batch: int = 512):
        # 결정 G3 (2026-10-23): CPU + 결정론 모드. 끄면 같은 시드에서도 조기 종료 시점이 갈려 결과가 달라진다.
        # MPS는 index_add_ 역전파에 결정론 구현이 없어 deterministic=True와 함께 쓸 수 없다 (docs/bench_gnn.md)
        if deterministic and device != "cpu":
            raise ValueError(f"deterministic=True는 CPU에서만 가능하다 (device={device})")
        self.task, self.params, self.seed = task, dict(params), seed
        self.X_va, self.y_va = valid
        self.max_epochs, self.patience, self.device, self.eval_batch = max_epochs, patience, device, eval_batch
        self.deterministic = deterministic
        self.history: list[float] = []
        self.best_epoch = -1

    def _new_model(self) -> DMPNN:
        p = self.params
        return DMPNN(d_h=p["d_h"], depth=p["depth"], dropout=p["dropout"], ffn_hidden=p["ffn_hidden"],
                     ffn_layers=p["ffn_layers"]).to(self.device)

    def _raw(self, X: Sequence[MolGraph]) -> np.ndarray:
        self.model.eval()
        out = []
        with torch.no_grad(), _deterministic(self.deterministic):
            for i in range(0, len(X), self.eval_batch):
                out.append(self.model(batch_graphs(X[i:i + self.eval_batch]).to(self.device)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def _valid_score(self) -> float:
        """조기 종료 기준: 회귀 RMSE(작을수록), 분류 AUPRC(클수록) → 항상 '클수록 좋음'으로 반환."""
        if self.task == "regression":
            return -regression_metrics(self.y_va, self.predict(self.X_va))["rmse"]
        s = classification_metrics(self.y_va, self.predict_proba(self.X_va)[:, 1], threshold=0.5)["auprc"]
        return -np.inf if np.isnan(s) else s

    def fit(self, X: Sequence[MolGraph], y) -> DMPNNEstimator:
        with _deterministic(self.deterministic):
            return self._fit(X, y)

    def _fit(self, X: Sequence[MolGraph], y) -> DMPNNEstimator:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y, dtype=np.float64)
        # 회귀는 학습셋 기준 표준화 (chemprop과 같은 관행)
        self.mu, self.sigma = (y.mean(), y.std() or 1.0) if self.task == "regression" else (0.0, 1.0)
        target = torch.tensor((y - self.mu) / self.sigma, dtype=torch.float32)
        self.model = self._new_model()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])
        loss_fn = nn.MSELoss() if self.task == "regression" else nn.BCEWithLogitsLoss()
        bs = int(self.params["batch_size"])

        best, best_state, since = -np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.model.train()
            order = rng.permutation(len(X))
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                if len(idx) < 2 and len(order) > 1:  # 1개짜리 마지막 배치는 건너뛴다
                    continue
                g = batch_graphs([X[j] for j in idx]).to(self.device)
                loss = loss_fn(self.model(g), target[idx].to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            score = self._valid_score()
            self.history.append(score)
            if score > best:
                best, best_state, since, self.best_epoch = score, copy.deepcopy(self.model.state_dict()), 0, epoch
            else:
                since += 1
                if since >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: Sequence[MolGraph]) -> np.ndarray:
        raw = self._raw(X)
        return raw * self.sigma + self.mu if self.task == "regression" else (raw > 0).astype(int)

    def predict_proba(self, X: Sequence[MolGraph]) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self._raw(X)))
        return np.stack([1 - p, p], axis=1)
