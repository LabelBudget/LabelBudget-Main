"""학습 규칙 원인 진단 (04 §6.2): 대예산에서 chemprop 기본값이 C보다 나은 이유가 학습 규칙인가 HP 선택인가.

python scripts/analyze_chemprop_ablation.py --dataset aqsoldb
→ reports/chemprop_ablation_aqsoldb.md

설계: chemprop 기본값(HP·규칙 모두 기본)에서 C의 학습 규칙만 덧씌운다. HP는 기본값 그대로다.
- abl_constlr: Noam → 고정 lr 1e-3 / abl_mean: norm → mean readout / abl_patience: valid RMSE patience 10 조기 종료
- abl_allrules: 셋 다 + 최대 100에폭 (= C의 학습 규칙 전부, HP만 기본값)

판정 규칙 (결과를 보기 전에 정함)
- 격차 G = 예산별 중앙값(C − 기본값)
- 각 모드 − 기본값: 시드 쌍 Wilcoxon, 4모드 × 2예산 = 8개에 Holm
- allrules − 기본값이 Holm p < 0.05이고 중앙값 ≥ G/2 → "학습 규칙이 격차의 주원인".
  그 안에서 단일 요인 중 Holm p < 0.05이고 중앙값 > 0인 것을 원인 후보로 크기 순으로 적는다
- 그렇지 않으면 → "학습 규칙으로 설명되지 않음, HP 선택(탐색) 쪽"
- 보조: allrules − C (규칙은 같고 HP만 다름, 2예산 Holm). 차이가 없으면 탐색이 고른 HP는 문제가 아니다
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_baselines import budget_order, fmt  # noqa: E402
from analyze_chemprop import wilcoxon_by_budget  # noqa: E402

from labelbudget.evaluation.stats import holm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ABL = {"C_chemprop_abl_constlr": "고정 lr 1e-3", "C_chemprop_abl_mean": "mean readout",
       "C_chemprop_abl_patience": "patience 10 조기 종료", "C_chemprop_abl_allrules": "C 규칙 전부 (HP 기본값)"}
BUDGETS = ("3000", "full")


def load(dataset: str, split: str, n_trials: int) -> pd.DataFrame:
    res = ROOT / "reports" / "results"
    gnn = pd.read_csv(res / f"{dataset}_gnn_full.csv", dtype={"n_labels": str})
    gnn = gnn[gnn.hp_n_trials == n_trials]
    cp = pd.read_csv(res / f"{dataset}_chemprop_full.csv", dtype={"n_labels": str})
    cp = cp[cp.model == "C_chemprop_default"]
    abl = pd.read_csv(res / f"{dataset}_chemprop_ablation.csv", dtype={"n_labels": str})
    d = pd.concat([gnn, cp, abl])
    d = d[(d.metric_name == "rmse") & (d.split_strategy == split) & d.n_labels.isin(BUDGETS)]
    return d.pivot_table(index=["n_labels", "split_seed"], columns="model", values="metric_value")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="scaffold_random")
    p.add_argument("--n-trials", type=int, default=20)
    args = p.parse_args()

    w = load(args.dataset, args.split, args.n_trials)
    missing = w.isna().sum()
    if missing.any() or set(ABL) - set(w.columns):
        raise SystemExit(f"빠진 조합이 있다: {missing[missing > 0].to_dict()}, 없는 모드: {set(ABL) - set(w.columns)}")
    budgets = sorted(w.index.get_level_values(0).unique(), key=budget_order)
    gap = {n: float((w.loc[n].C_dmpnn - w.loc[n].C_chemprop_default).median()) for n in budgets}

    # 8개 검정에 한 번에 Holm (wilcoxon_by_budget은 예산 간 Holm이라 원 p만 쓴다)
    tests = [{"model": m, **r} for m in ABL for r in wilcoxon_by_budget(w, m, "C_chemprop_default")]
    for r, ph in zip(tests, holm([r["p_value"] for r in tests])):
        r["p_holm8"] = ph
    t = {(r["model"], r["n_labels"]): r for r in tests}

    md = [f"# 학습 규칙 원인 진단 — {args.dataset}", "",
          f"- 분할 {args.split}, 시드 {w.index.get_level_values(1).nunique()}개, 예산 {', '.join(budgets)}, test RMSE",
          "- chemprop 기본값에서 C의 학습 규칙만 바꿨다. HP(hidden 300, depth 3, FFN 300×1, dropout 0, batch 64)는 기본값",
          "- 판정 규칙은 `scripts/analyze_chemprop_ablation.py` 머리말에 실행 전에 적었다", "",
          "## 1. RMSE (평균 ± SD / 중앙값)", "",
          "| 모드 | " + " | ".join(budgets) + " |", "|---|" + "---:|" * len(budgets)]
    rows = {"C_chemprop_default": "chemprop 기본값", **ABL, "C_dmpnn": "C 직접 구현 (탐색 20회)"}
    md += [f"| {name} | " + " | ".join(f"{fmt(w.loc[n][m].mean())} ± {fmt(w.loc[n][m].std())} / "
                                       f"{fmt(w.loc[n][m].median())}" for n in budgets) + " |"
           for m, name in rows.items()]
    md += ["", "## 2. 기본값 대비 차이 (양수 = 그 규칙이 나쁘게 만든다)", "",
           "| 모드 | 라벨 수 | 중앙값 차이 | 나빠진 시드 | p | p (Holm, 8개) | 격차 G 대비 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for m, name in ABL.items():
        for n in budgets:
            r, x = t[(m, n)], w.loc[n]
            md.append(f"| {name} | {n} | {fmt(r['median_diff'])} | {int((x[m] > x.C_chemprop_default).sum())}/{len(x)} | "
                      f"{fmt(r['p_value'], 4)} | {fmt(r['p_holm8'], 4)} | {r['median_diff'] / gap[n]:.0%} |")
    md += ["", "격차 G = 중앙값(C − 기본값): " + ", ".join(f"{n} {fmt(g)}" for n, g in gap.items()), ""]

    aux = wilcoxon_by_budget(w, "C_chemprop_abl_allrules", "C_dmpnn")
    md += ["## 3. 보조: C 규칙 전부(HP 기본값) − C (규칙 같음, HP만 다름)", "",
           "| 라벨 수 | 중앙값 차이 | p (Holm) |", "|---:|---:|---:|"]
    md += [f"| {r['n_labels']} | {fmt(r['median_diff'])} | {fmt(r['p_holm'], 4)} |" for r in aux]
    md.append("")

    md += ["## 4. 판정", ""]
    for n in budgets:
        a = t[("C_chemprop_abl_allrules", n)]
        if a["p_holm8"] < 0.05 and a["median_diff"] >= gap[n] / 2:
            causes = sorted([(t[(m, n)]["median_diff"], ABL[m]) for m in ABL if m != "C_chemprop_abl_allrules"
                             and t[(m, n)]["p_holm8"] < 0.05 and t[(m, n)]["median_diff"] > 0], reverse=True)
            cause = ", ".join(f"{name} (+{fmt(v)})" for v, name in causes) or "단일 요인으로는 유의하지 않음 (조합 효과)"
            md.append(f"- **{n}**: 학습 규칙이 격차의 주원인 (C 규칙 전부 +{fmt(a['median_diff'])}, G {fmt(gap[n])}). "
                      f"원인 후보: {cause}")
        else:
            md.append(f"- **{n}**: 학습 규칙으로 설명되지 않음 (C 규칙 전부 {fmt(a['median_diff'])}, "
                      f"Holm p {fmt(a['p_holm8'], 4)}, G {fmt(gap[n])}) → HP 선택(탐색) 쪽")
    md.append("")

    out = ROOT / "reports" / f"chemprop_ablation_{args.dataset}.md"
    out.write_text("\n".join(md))
    print(f"→ {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
