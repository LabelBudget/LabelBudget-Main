"""RDKit 2D 서술자. 학습셋 기준 컬럼 고정. 스펙 02 §7.2.

전체 분자에 대해 한 번 계산하고(라벨 무관이라 누수 없음), 컬럼 선택은 학습셋에서만 정한다.
NaN은 LightGBM이 직접 처리하므로 대치하지 않는다.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

MAX_ABS = 1e10  # 이 이상은 발산으로 보고 NaN 처리


def compute_descriptors(smiles: Sequence[str]) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"SMILES 파싱 실패 (행 {i}): {s}")
        rows.append(Descriptors.CalcMolDescriptors(mol, missingVal=np.nan))
    df = pd.DataFrame(rows).astype(float)
    return df.mask(~np.isfinite(df) | (df.abs() > MAX_ABS))


@dataclass
class DescriptorSelector:
    """학습셋에서 상수 컬럼·결측 과다 컬럼을 제거하고 그 목록을 고정한다."""
    max_missing_frac: float = 0.2
    columns: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)
    dropped_missing: list[str] = field(default_factory=list)

    def fit(self, X_train: pd.DataFrame) -> DescriptorSelector:
        missing = X_train.isna().mean()
        self.dropped_missing = missing.index[missing > self.max_missing_frac].tolist()
        rest = X_train.drop(columns=self.dropped_missing)
        self.dropped_constant = rest.columns[rest.nunique(dropna=True) <= 1].tolist()
        self.columns = [c for c in rest.columns if c not in self.dropped_constant]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.columns:
            raise RuntimeError("fit을 먼저 호출하세요")
        return X[self.columns].to_numpy(dtype=np.float32)

    def summary(self) -> dict:
        return {"n_used": len(self.columns), "n_dropped_constant": len(self.dropped_constant),
                "n_dropped_missing": len(self.dropped_missing)}
