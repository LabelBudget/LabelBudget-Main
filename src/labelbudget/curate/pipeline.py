"""CurationLedger와 render_data_card. 스펙 02 §5.

중복을 발생 단계별로 나눠 집계한다:
  incoming      — 입력 InChIKey가 이미 같았던 중복 (들어온 중복). 표준 InChI의 이동 수소 처리 포함
  standardized  — 표준화가 만든 중복. 그중 토토머 정규화가 관여한 것을 tautomer로 따로 센다
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from labelbudget.curate.dedupe import ConflictRule, deduplicate
from labelbudget.curate.standardize import StandardizeConfig, Standardizer, StandardizeResult

IMBALANCE_WARN = 5.0


@dataclass
class CurationLedger:
    dataset: str
    task: str
    label_name: str
    label_unit: str
    created_utc: str
    n_raw: int = 0
    n_after_missing: int = 0
    n_after_standardize: int = 0
    n_after_dedupe: int = 0
    reject_reasons: dict = field(default_factory=dict)
    standardize_notes: dict = field(default_factory=dict)
    n_structure_changed: int = 0
    n_duplicates_incoming: int = 0
    n_duplicates_created_by_standardization: int = 0
    n_duplicates_created_by_tautomer: int = 0
    duplicate_label_spread_stats: dict = field(default_factory=dict)
    tautomer_duplicate_examples: list = field(default_factory=list)
    dedupe: dict = field(default_factory=dict)
    standardize_config: dict = field(default_factory=dict)
    label_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _spread_stats(spreads: pd.Series, task: str) -> dict:
    if spreads.empty:
        return {"n_groups": 0}
    if task == "classification":
        return {"n_groups": int(len(spreads)), "n_label_disagree": int((spreads > 0).sum())}
    return {
        "n_groups": int(len(spreads)),
        "mean": float(spreads.mean()),
        "median": float(spreads.median()),
        "max": float(spreads.max()),
        **{f"n_gt_{t}": int((spreads > t).sum()) for t in (0.5, 1.0, 2.0)},
    }


def label_stats(y: pd.Series, task: str) -> dict:
    if task == "classification":
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        return {"n": int(len(y)), "n_positive": n_pos, "n_negative": n_neg,
                "positive_rate": n_pos / len(y) if len(y) else float("nan"),
                "imbalance_ratio": n_neg / n_pos if n_pos else float("inf")}
    q = y.quantile([0.25, 0.5, 0.75])
    return {"n": int(len(y)), "mean": float(y.mean()), "std": float(y.std()), "min": float(y.min()),
            "p25": float(q[0.25]), "median": float(q[0.5]), "p75": float(q[0.75]), "max": float(y.max())}


def standardize_frame(
    df: pd.DataFrame, std: Standardizer, keep_cols: tuple[str, ...] = ()
) -> tuple[pd.DataFrame, list[StandardizeResult]]:
    """`smiles`, `label` 입력을 표준화한다. 통과 행 DataFrame과 전체 결과 목록을 반환한다."""
    results = [std(s) for s in df.smiles]
    ok = np.array([r.ok for r in results], dtype=bool)
    passed = pd.DataFrame({
        "smiles": [r.smiles for r in results if r.ok],
        "smiles_input": df.smiles[ok].tolist(),
        "inchikey": [r.inchikey for r in results if r.ok],
        "input_inchikey": [r.input_inchikey for r in results if r.ok],
        "pre_tautomer_inchikey": [r.pre_tautomer_inchikey for r in results if r.ok],
        "label": df.label[ok].tolist(),
    })
    for c in keep_cols:
        passed[c] = df[c][ok].tolist()
    return passed, results


def duplicate_groups(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """최종 키 기준 중복 그룹을 한 행씩: key, n, spread, origin.

    origin — incoming: 입력 키가 이미 같았음 / tautomer: 토토머 정규화가 관여 / standardized: 그 외 표준화 단계
    """
    rows = []
    for key, g in df.groupby("inchikey", sort=False):
        if len(g) < 2:
            continue
        # 입력 키를 못 만든 행은 각자 고유로 본다
        input_keys = g["input_inchikey"].fillna(pd.Series([f"__na_{i}" for i in g.index], index=g.index))
        if input_keys.nunique() == 1:
            origin = "incoming"
        elif g["pre_tautomer_inchikey"].nunique(dropna=False) > 1:
            origin = "tautomer"
        else:
            origin = "standardized"
        sp = float(g.label.max() - g.label.min()) if task == "regression" else float(g.label.nunique() - 1)
        rows.append({"key": key, "n": len(g), "spread": sp, "origin": origin, "index": g.index.tolist()})
    return pd.DataFrame(rows, columns=["key", "n", "spread", "origin", "index"])


def _duplicate_origins(df: pd.DataFrame, task: str) -> tuple[dict, list]:
    """원장용 집계. standardized 수에는 tautomer가 포함된다."""
    groups = duplicate_groups(df, task)
    created = groups[groups.origin != "incoming"]
    taut = groups[groups.origin == "tautomer"]
    counts = {"incoming": int((groups.origin == "incoming").sum()),
              "standardized": len(created), "tautomer": len(taut)}
    stats = {"standardized": _spread_stats(created.spread, task), "tautomer": _spread_stats(taut.spread, task)}
    examples = [{"key": r.key, "spread": r.spread, "labels": df.label[r.index].tolist(),
                 "smiles_input": df.smiles_input[r.index].tolist()}
                for r in taut.sort_values("spread", ascending=False).itertuples()]
    return {"counts": counts, "stats": stats}, examples


def curate(
    df: pd.DataFrame,
    *,
    dataset: str,
    task: str,
    label_name: str,
    label_unit: str,
    standardize_config: StandardizeConfig | None = None,
    conflict_rule: ConflictRule = ConflictRule.DROP,
    conflict_tolerance: float = 0.5,
    keep_cols: tuple[str, ...] = (),
    max_examples: int = 20,
) -> tuple[pd.DataFrame, CurationLedger]:
    """`smiles`, `label` (+ keep_cols) 입력 → 큐레이션 결과와 원장."""
    cfg = standardize_config or StandardizeConfig()
    ledger = CurationLedger(dataset=dataset, task=task, label_name=label_name, label_unit=label_unit,
                            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            standardize_config=cfg.to_dict())
    ledger.n_raw = len(df)

    work = df.dropna(subset=["smiles", "label"]).reset_index(drop=True)
    ledger.n_after_missing = len(work)

    passed, results = standardize_frame(work, Standardizer(cfg), keep_cols)
    ledger.reject_reasons = dict(Counter(r.reason for r in results if not r.ok).most_common())
    ledger.standardize_notes = dict(Counter(n for r in results if r.ok for n in r.notes).most_common())
    ledger.n_structure_changed = sum(r.changed for r in results if r.ok)
    ledger.n_after_standardize = len(passed)

    origins, taut_examples = _duplicate_origins(passed, task)
    ledger.n_duplicates_incoming = origins["counts"]["incoming"]
    ledger.n_duplicates_created_by_standardization = origins["counts"]["standardized"]
    ledger.n_duplicates_created_by_tautomer = origins["counts"]["tautomer"]
    ledger.duplicate_label_spread_stats = origins["stats"]
    ledger.tautomer_duplicate_examples = taut_examples[:max_examples]

    out, report = deduplicate(passed, task=task, rule=conflict_rule, tolerance=conflict_tolerance,
                              max_examples=max_examples)
    ledger.dedupe = report.to_dict()
    ledger.n_after_dedupe = len(out)
    ledger.label_stats = label_stats(out.label, task)
    out = out.drop(columns=["input_inchikey", "pre_tautomer_inchikey"])
    return out, ledger


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _spread_table(stats: dict) -> list[str]:
    if not stats or stats.get("n_groups", 0) == 0:
        return ["없음."]
    return ["| 항목 | 값 |", "|---|---:|", *[f"| {k} | {_fmt(v)} |" for k, v in stats.items()]]


def render_data_card(ledger: CurationLedger) -> str:
    L = ledger
    d = L.dedupe
    lines = [f"# 데이터 카드 — {L.dataset}", "",
             f"- 과제: {L.task} · 라벨: `{L.label_name}` ({L.label_unit})",
             f"- 생성: {L.created_utc}", ""]

    # 1. 단계별 생존
    steps = [("원본", L.n_raw), ("결측 제거", L.n_after_missing),
             ("표준화", L.n_after_standardize), ("중복 제거", L.n_after_dedupe)]
    lines += ["## 1. 단계별 생존 개수", "", "| 단계 | 개수 | 직전 대비 |", "|---|---:|---:|"]
    prev = None
    for name, n in steps:
        lines.append(f"| {name} | {n:,} | {'' if prev is None else f'{n - prev:+,}'} |")
        prev = n
    keep = L.n_after_dedupe / L.n_raw * 100 if L.n_raw else 0.0
    lines += ["", f"최종 보존율 **{keep:.1f}%**", ""]

    # 2. 거부 사유
    lines += ["## 2. 표준화 거부 사유", ""]
    if L.reject_reasons:
        lines += ["| 사유 | 개수 |", "|---|---:|", *[f"| `{k}` | {v:,} |" for k, v in L.reject_reasons.items()]]
    else:
        lines.append("없음.")
    lines.append("")

    # 3. 구조 변경
    lines += ["## 3. 구조 변경 내역", "", f"SMILES가 바뀐 분자: **{L.n_structure_changed:,}**", ""]
    if L.standardize_notes:
        lines += ["| 변경 유형 | 개수 |", "|---|---:|", *[f"| `{k}` | {v:,} |" for k, v in L.standardize_notes.items()]]
    lines.append("")

    # 4. 중복과 충돌
    tol = "—" if d["conflict_tolerance"] is None else d["conflict_tolerance"]
    lines += ["## 4. 중복 및 라벨 충돌", "",
              "| 항목 | 값 |", "|---|---:|",
              f"| 완전 중복 행 | {d['n_exact_duplicate_rows']:,} |",
              f"| 고유 분자 | {d['n_unique_molecules']:,} |",
              f"| 충돌 분자 | {d['n_conflicting_molecules']:,} |",
              f"| 충돌로 제거 | {d['n_dropped_by_conflict']:,} |",
              f"| 규칙 / tolerance | `{d['conflict_rule']}` / {tol} |",
              f"| 골격 충돌 그룹 / 분자 | {d['n_skeleton_collision_groups']:,} / {d['n_skeleton_collision_molecules']:,} |", ""]
    if d["n_skeleton_collision_groups"]:
        lines += ["> ⚠️ 골격(InChIKey 첫 블록)이 같은 분자는 제거하지 않았다. "
                  "train/test로 갈라지면 누수가 되므로 분할 후 `leakage_check`로 점검한다.", ""]
    if d["examples"]:
        lines += ["충돌 상위 예시:", "", "| key | n | 라벨 | spread | 제거 |", "|---|---:|---|---:|---|"]
        lines += [f"| `{e['key']}` | {e['n']} | {', '.join(_fmt(float(v)) for v in e['labels'])} | "
                  f"{e['spread']:.3f} | {'예' if e['dropped'] else '아니오'} |" for e in d["examples"]]
        lines.append("")

    # 5. 표준화가 만든 중복
    lines += ["## 5. 표준화가 만들어낸 중복", "",
              "| 발생 단계 | 그룹 수 |", "|---|---:|",
              f"| 들어온 중복 (입력 InChIKey 동일) | {L.n_duplicates_incoming:,} |",
              f"| 표준화가 만든 중복 | {L.n_duplicates_created_by_standardization:,} |",
              f"| └ 그중 토토머 정규화가 관여 | {L.n_duplicates_created_by_tautomer:,} |", "",
              "> 표준 InChI는 이동 수소 토토머(예: 2-하이드록시피리딘/2-피리돈)를 이미 같은 키로 보므로 "
              "그런 쌍은 '들어온 중복'에 집계된다. '토토머 정규화가 관여'는 RDKit 규칙으로만 합쳐진 것(예: 케토-엔올)이다.", "",
              "### 표준화가 만든 중복의 라벨 불일치", "",
              *_spread_table(L.duplicate_label_spread_stats.get("standardized", {})), "",
              "### 토토머 정규화가 관여한 중복", "",
              *_spread_table(L.duplicate_label_spread_stats.get("tautomer", {})), ""]
    if L.tautomer_duplicate_examples:
        lines += ["> 토토머 규칙은 이중결합 위치 이성질체처럼 **서로 다른 화합물**을 합칠 수 있다. "
                  "아래 예시를 확인하고, spread가 tolerance 이하라 병합된 그룹은 한계로 기록한다.", "",
                  "| spread | 라벨 | 입력 SMILES |", "|---:|---|---|"]
        lines += [f"| {e['spread']:.3f} | {', '.join(_fmt(float(v)) for v in e['labels'])} | "
                  f"{'<br>'.join(f'`{s}`' for s in e['smiles_input'])} |" for e in L.tautomer_duplicate_examples]
        lines.append("")

    # 6. 라벨 분포
    lines += ["## 6. 라벨 분포", "", "| 통계 | 값 |", "|---|---:|",
              *[f"| {k} | {_fmt(v)} |" for k, v in L.label_stats.items()], ""]
    ratio = L.label_stats.get("imbalance_ratio")
    if ratio is not None and ratio > IMBALANCE_WARN:
        lines += [f"> ⚠️ 음성:양성 = {ratio:.1f}:1 로 불균형이 심하므로 AUROC 단독 보고는 낙관적이다. "
                  "AUPRC와 PPV/NPV를 병기한다.", ""]

    # 7. 설정
    lines += ["## 7. 표준화 설정", "", "```json", json.dumps(L.standardize_config, indent=2, ensure_ascii=False), "```", "",
              "순서 고정: 파싱 → Cleanup → [염 제거 | 다단편 거부] → Uncharger → 토토머 → 입체 → 필터. "
              "순서가 바뀌면 결과가 바뀐다.", ""]
    return "\n".join(lines)
