"""ECFP4 (Morgan r=2, 2048bit). 스펙 02 §7.1."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def ecfp4(smiles: Sequence[str], *, n_bits: int = 2048, radius: int = 2, counts: bool = False) -> np.ndarray:
    """(n, n_bits) uint8 배열. 파싱 실패는 예외 — 큐레이션을 통과한 SMILES만 받는다."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"SMILES 파싱 실패 (행 {i}): {s}")
        fp = gen.GetCountFingerprintAsNumPy(mol) if counts else gen.GetFingerprintAsNumPy(mol)
        out[i] = np.minimum(fp, 255)
    return out
