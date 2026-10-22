"""InChIKey 완전 중복·골격 중복, 라벨 충돌, 누수 점검. 스펙 02 §4.

충돌은 병합 대상이 아니라 거부 대상이 기본이다 (03 §4).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd


class ConflictRule(str, Enum):
    DROP = "drop"          # 충돌 건 전부 제거 (회귀 기본값)
    MEDIAN = "median"      # 회귀: 중앙값
    MEAN = "mean"          # 회귀: 평균
    MAJORITY = "majority"  # 분류: 다수결, 동수면 제거


@dataclass
class DedupeReport:
    n_input: int
    n_output: int
    n_exact_duplicate_rows: int
    n_unique_molecules: int
    n_conflicting_molecules: int
    n_dropped_by_conflict: int
    n_skeleton_collision_groups: int
    n_skeleton_collision_molecules: int
    conflict_rule: str
    conflict_tolerance: float | None
    examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _skeleton(inchikey: str) -> str:
    return inchikey.split("-")[0]


def _validate_rule(task: str, rule: ConflictRule) -> None:
    if task == "regression" and rule is ConflictRule.MAJORITY:
        raise ValueError("MAJORITY는 분류 전용 규칙입니다")
    if task == "classification" and rule in (ConflictRule.MEDIAN, ConflictRule.MEAN):
        raise ValueError(f"{rule.value}는 회귀 전용 규칙입니다")


def skeleton_collisions(keys: Iterable[str]) -> tuple[int, int]:
    """골격(첫 14자)이 같고 전체 키가 다른 그룹 수, 그 그룹에 속한 분자 수."""
    s = pd.Series(sorted(set(keys)))
    sizes = s.map(_skeleton).value_counts()
    sizes = sizes[sizes > 1]
    return int(len(sizes)), int(sizes.sum())


def deduplicate(
    df: pd.DataFrame,
    *,
    key_col: str = "inchikey",
    label_col: str = "label",
    task: Literal["regression", "classification"] = "regression",
    rule: ConflictRule = ConflictRule.DROP,
    tolerance: float = 0.5,
    max_examples: int = 20,
) -> tuple[pd.DataFrame, DedupeReport]:
    """키 기준으로 한 행씩 남긴다. 비라벨 컬럼은 첫 행 값을 쓰고 병합 행 수를 `n_merged`에 남긴다.

    회귀: 그룹 내 max-min > tolerance 이면 충돌. 남는 그룹은 중앙값(MEAN 규칙이면 평균)으로 합친다.
    분류: 라벨 종류가 2개 이상이면 충돌 (tolerance 무시).
    """
    rule = ConflictRule(rule)
    _validate_rule(task, rule)
    if df[key_col].isna().any():
        raise ValueError(f"'{key_col}'에 결측이 있습니다. 표준화 실패 행을 먼저 제거하세요")
    if df[label_col].isna().any():
        raise ValueError(f"'{label_col}'에 결측이 있습니다. 결측 라벨을 먼저 제거하세요")

    work = df.reset_index(drop=True)
    groups = work.groupby(key_col, sort=False)[label_col]
    size = groups.transform("size")
    if task == "regression":
        spread = groups.transform(lambda v: v.max() - v.min())
        conflict = spread > tolerance
    else:
        spread = groups.transform("nunique").astype(float) - 1.0
        conflict = spread > 0

    if task == "regression":
        keep_label = groups.transform("mean" if rule is ConflictRule.MEAN else "median")
    else:
        keep_label = work[label_col].copy()
    drop_mask = pd.Series(False, index=work.index)
    if conflict.any():
        c = conflict
        if rule is ConflictRule.DROP:
            drop_mask = c
        elif rule is ConflictRule.MAJORITY:
            def majority(v: pd.Series) -> float:
                counts = v.value_counts()
                return np.nan if len(counts) > 1 and counts.iloc[0] == counts.iloc[1] else counts.index[0]
            maj = groups.transform(majority)
            keep_label[c] = maj[c]
            drop_mask = c & maj.isna()

    examples = []
    if conflict.any():
        conf = work[conflict].assign(_spread=spread[conflict])
        for key, g in sorted(conf.groupby(key_col, sort=False), key=lambda kv: -kv[1]["_spread"].iloc[0])[:max_examples]:
            ex = {"key": key, "n": len(g), "labels": g[label_col].tolist(),
                  "spread": float(g["_spread"].iloc[0]), "dropped": bool(drop_mask[g.index].all())}
            if "smiles" in g:
                ex["smiles"] = g["smiles"].iloc[0]
            examples.append(ex)

    out = work.assign(**{label_col: keep_label, "n_merged": size})[~drop_mask]
    out = out.drop_duplicates(subset=key_col, keep="first").reset_index(drop=True)
    if task == "classification":
        out[label_col] = out[label_col].astype(work[label_col].dtype)

    n_unique = int(work[key_col].nunique())
    n_skel_groups, n_skel_mols = skeleton_collisions(out[key_col])
    report = DedupeReport(
        n_input=len(work),
        n_output=len(out),
        n_exact_duplicate_rows=len(work) - n_unique,
        n_unique_molecules=n_unique,
        n_conflicting_molecules=int(work.loc[conflict, key_col].nunique()),
        n_dropped_by_conflict=int(work.loc[drop_mask, key_col].nunique()),
        n_skeleton_collision_groups=n_skel_groups,
        n_skeleton_collision_molecules=n_skel_mols,
        conflict_rule=rule.value,
        conflict_tolerance=tolerance if task == "regression" else None,
        examples=examples,
    )
    return out, report


def leakage_check(train_keys: Iterable[str], test_keys: Iterable[str]) -> dict:
    """exact_overlap: 양쪽에 있는 고유 키 수. skeleton_overlap: 키는 다르지만 골격이 train에 있는 test 고유 키 수.

    exact_overlap > 0 이면 실험을 중단해야 한다. skeleton_overlap은 경고로 보고한다.
    """
    train, test = set(train_keys), set(test_keys)
    exact = train & test
    train_skel = {_skeleton(k) for k in train}
    skel = sum(1 for k in test - exact if _skeleton(k) in train_skel)
    return {"exact_overlap": len(exact), "skeleton_overlap": skel, "clean": not exact}
