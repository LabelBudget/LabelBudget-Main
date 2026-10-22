"""데이터셋 감사 — 염 처리 정책 A/B 비교와 노이즈 하한. 스펙 02 §13.

정책 A: strip_salts=True (염 제거) / 정책 B: strip_salts=False (다단편 거부).
각 정책을 토토머 정규화 포함(스펙 설정)·미포함(03 문서 조건)으로 돌린다.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rdkit import Chem

from labelbudget.curate.pipeline import _spread_stats, duplicate_groups, label_stats, standardize_frame
from labelbudget.curate.standardize import StandardizeConfig, Standardizer, _safe_inchikey
from labelbudget.data.loaders import load_raw
from labelbudget.data.sources import get_source

POLICIES: dict[str, StandardizeConfig] = {
    "A_strip": StandardizeConfig(strip_salts=True),
    "B_reject": StandardizeConfig(strip_salts=False),
    "A_strip_no_taut": StandardizeConfig(strip_salts=True, canonicalize_tautomer=False),
    "B_reject_no_taut": StandardizeConfig(strip_salts=False, canonicalize_tautomer=False),
}


@dataclass
class PolicyResult:
    name: str
    config: dict
    n_input: int
    n_pass: int
    n_unique: int
    reject_reasons: dict
    n_duplicate_groups: int          # 최종 키 기준 전체 중복 그룹
    n_created_groups: int            # 그중 표준화가 만든 것
    n_created_rows: int
    created_spread: dict
    top_conflicts: list = field(default_factory=list)


def run_policy(df: pd.DataFrame, name: str, cfg: StandardizeConfig, task: str, top_n: int = 10) -> PolicyResult:
    passed, results = standardize_frame(df, Standardizer(cfg), keep_cols=("name",))
    groups = duplicate_groups(passed, task)
    created = groups[groups.origin != "incoming"]
    top = []
    for r in created.sort_values("spread", ascending=False).head(top_n).itertuples():
        members = passed.loc[r.index].sort_values("label", ascending=False)
        top.append({"key": r.key, "spread": r.spread, "n": r.n,
                    "members": [{"name": m.name, "label": float(m.label), "smiles": m.smiles_input}
                                for m in members.itertuples()]})
    return PolicyResult(
        name=name,
        config=cfg.to_dict(),
        n_input=len(df),
        n_pass=len(passed),
        n_unique=int(passed.inchikey.nunique()),
        reject_reasons=dict(Counter(r.reason for r in results if not r.ok).most_common()),
        n_duplicate_groups=len(groups),
        n_created_groups=len(created),
        n_created_rows=int(created.n.sum()),
        created_spread=_spread_stats(created.spread, task),
        top_conflicts=top,
    )


def _noise_floor(sd: pd.Series) -> dict:
    pos = sd[sd > 0]
    return {
        "n_sd_zero": int((sd == 0).sum()),
        "n_sd_positive": int(len(pos)),
        "mean": float(pos.mean()),
        "median": float(pos.median()),
        "rms": float(np.sqrt((pos ** 2).mean())),
        "max": float(pos.max()),
        **{f"n_gt_{t}": int((pos > t).sum()) for t in (0.5, 1.0, 2.0)},
    }


def audit(dataset: str, label: str | None = None, *, policies: list[str] | None = None,
          top_n: int = 10, raw_dir=None) -> dict:
    src = get_source(dataset)
    raw = load_raw(dataset, raw_dir, allow_download=False)
    label = label or src.default_label or src.label_cols[0]
    if label not in src.label_cols:
        raise ValueError(f"'{dataset}' 라벨 후보: {list(src.label_cols)}")

    report: dict = {"dataset": dataset, "task": src.task, "label": label,
                    "shape": list(raw.shape), "columns": list(raw.columns)}

    df = pd.DataFrame({
        "smiles": raw[src.smiles_col],
        "label": raw[label],
        "name": raw[src.name_col] if src.name_col else pd.Series(range(len(raw))).map(lambda i: f"row{i}"),
    })
    report["n_missing_label"] = int(df.label.isna().sum())
    df = df.dropna(subset=["smiles", "label"]).reset_index(drop=True)
    report["label_stats"] = label_stats(df.label, src.task)

    extra = {}
    if "SD" in raw:
        extra["noise_floor"] = _noise_floor(raw["SD"])
    if "Ocurrences" in raw:
        extra["ocurrences"] = {str(k): int(v) for k, v in raw["Ocurrences"].value_counts().sort_index().items()}
    if "Group" in raw:
        extra["group"] = {str(k): int(v) for k, v in raw["Group"].value_counts().sort_index().items()}
    report["extra"] = extra

    keys = [_safe_inchikey(m) if (m := Chem.MolFromSmiles(s)) is not None else None for s in raw[src.smiles_col]]
    keys = pd.Series(keys).dropna()
    report["raw_keys"] = {"n_parsed": int(len(keys)), "n_unique_inchikey": int(keys.nunique()),
                          "n_unique_skeleton": int(keys.str.split("-").str[0].nunique())}
    if "InChIKey" in raw:
        file_keys = raw["InChIKey"].dropna()
        report["raw_keys"]["file_inchikey_unique"] = int(file_keys.nunique())
        report["raw_keys"]["file_skeleton_unique"] = int(file_keys.str.split("-").str[0].nunique())

    report["policies"] = {name: run_policy(df, name, POLICIES[name], src.task, top_n)
                          for name in (policies or list(POLICIES))}
    return report


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def render_audit(report: dict) -> str:
    R = report
    regression = R["task"] == "regression"
    lines = [f"# 감사 보고서 — {R['dataset']}", "",
             "> `scripts/audit_dataset.py`가 생성. 문서와 숫자가 어긋나면 이 출력을 신뢰한다.", "",
             "## 1. 파일", "", f"- shape: {R['shape'][0]:,} × {R['shape'][1]}",
             f"- 컬럼: {', '.join(f'`{c}`' for c in R['columns'])}", ""]

    lines += [f"## 2. 라벨 분포 (`{R['label']}`)", "", f"결측 라벨: {R['n_missing_label']:,}", "",
              "| 통계 | 값 |", "|---|---:|", *[f"| {k} | {_fmt(v)} |" for k, v in R["label_stats"].items()], ""]

    ex = R["extra"]
    lines += ["## 3. 신뢰도 컬럼", ""]
    if not ex:
        lines += ["해당 컬럼 없음.", ""]
    for col in ("ocurrences", "group"):
        if col in ex:
            lines += [f"`{col}` 분포: " + " / ".join(f"{k}: {v:,}" for k, v in ex[col].items()), ""]

    rk = R["raw_keys"]
    lines += ["## 4. 원본 InChIKey", "", "| 항목 | 값 |", "|---|---:|",
              *[f"| {k} | {v:,} |" for k, v in rk.items()], ""]

    pols = R["policies"]
    names = list(pols)
    lines += ["## 5. 염 처리 정책 비교", "",
              "A = 염 제거(`strip_salts=True`), B = 다단편 거부(`strip_salts=False`). "
              "`no_taut`는 토토머 정규화 미포함(03 문서 측정 조건).", "",
              "| 지표 | " + " | ".join(f"`{n}`" for n in names) + " |",
              "|---|" + "---:|" * len(names)]
    rows = [("통과", lambda p: p.n_pass), ("고유 InChIKey", lambda p: p.n_unique),
            ("중복 그룹 (전체)", lambda p: p.n_duplicate_groups),
            ("표준화가 만든 중복 그룹", lambda p: p.n_created_groups),
            ("└ 관련 행", lambda p: p.n_created_rows)]
    stat_keys = ["median", "max", "n_gt_0.5", "n_gt_1.0", "n_gt_2.0"] if regression else ["n_label_disagree"]
    rows += [(f"만든 중복 spread {k}", lambda p, k=k: p.created_spread.get(k, "—")) for k in stat_keys]
    lines += [f"| {label} | " + " | ".join(_fmt(f(pols[n])) for n in names) + " |" for label, f in rows]
    lines += ["", "거부 사유:", ""]
    lines += [f"- `{n}`: " + ", ".join(f"{k} {v:,}" for k, v in pols[n].reject_reasons.items()) for n in names]
    lines.append("")

    lines += ["## 6. 표준화가 만든 중복 — 상위 예시", ""]
    for n in names:
        if not pols[n].top_conflicts:
            continue
        lines += [f"### `{n}`", ""]
        for c in pols[n].top_conflicts:
            lines.append(f"- spread **{c['spread']:.2f}** (n={c['n']})")
            lines += [f"    - {m['label']:7.2f}  {m['name']}  `{m['smiles']}`" for m in c["members"][:6]]
            if len(c["members"]) > 6:
                lines.append(f"    - … 외 {len(c['members']) - 6}건")
        lines.append("")

    lines += ["## 7. 측정 노이즈 하한", ""]
    nf = ex.get("noise_floor")
    if nf:
        lines += ["| 항목 | 값 |", "|---|---:|", *[f"| {k} | {_fmt(v)} |" for k, v in nf.items()], "",
                  f"복수 출처 분자 {nf['n_sd_positive']:,}건의 출처 간 SD 평균 **{nf['mean']:.3f}**, "
                  f"RMS **{nf['rms']:.3f}**. RMSE와 비교할 척도는 RMS가 더 맞다.", ""]
    else:
        lines += ["`SD` 컬럼 없음 — 추정 불가.", ""]
    return "\n".join(lines)
