"""사후 민감도: 특정 분자를 test에서 뺐을 때 C − B 결론이 바뀌는가. LIMITATIONS '붕소 케이지' 항목.

python scripts/analyze_exclusion.py --dataset aqsoldb --exclude-smiles B12B3B4B1C234
→ reports/sensitivity_exclusion_aqsoldb.md

주 결과는 바꾸지 않는다. 분자를 test에 가진 시드만 --save-preds로 다시 돌려(C·B) 다음을 확인한다.
1. 재실행 지표 = 주 그리드 지표 (결정론 확인. 다르면 예측을 쓸 수 없으므로 멈춘다)
2. 저장한 예측으로 계산한 RMSE = CSV의 RMSE
3. 분자 제외 RMSE로 해당 시드 값을 바꾼 뒤 C − B Wilcoxon + Holm 재계산
HP 선택은 그대로 둔다 (분자가 valid에 있는 시드의 선택 효과는 측정하지 않는다).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_baselines import budget_order, fmt  # noqa: E402

from labelbudget.evaluation.stats import holm, paired_wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"C_dmpnn": "gnn", "B_desc_lgbm": "baselines"}
KEY = ["model", "split_strategy", "split_seed", "n_labels", "hp_n_trials"]


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--exclude-smiles", nargs="+", required=True)
    p.add_argument("--split", default="scaffold_random")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--rerun-tag", default="rerun_s6_s9", help="reports/results/sensitivity/<dataset>_<group>_<tag>.csv")
    args = p.parse_args()

    res = ROOT / "reports" / "results"
    df = pd.read_csv(ROOT / "data" / "processed" / f"{args.dataset}.csv", usecols=["smiles", "inchikey"])
    excl = set(df[df.smiles.isin(args.exclude_smiles)].inchikey)
    if len(excl) != len(args.exclude_smiles):
        raise SystemExit(f"큐레이션 데이터에서 찾지 못한 SMILES가 있다: {args.exclude_smiles}")

    main_rmse, rerun_rmse, pred_rows = [], [], []
    for model, group in MODELS.items():
        m = pd.read_csv(res / f"{args.dataset}_{group}_full.csv", dtype={"n_labels": str})
        main_rmse.append(m[(m.model == model) & (m.metric_name == "rmse") & (m.split_strategy == args.split)])
        r = pd.read_csv(res / "sensitivity" / f"{args.dataset}_{group}_{args.rerun_tag}.csv", dtype={"n_labels": str})
        rerun_rmse.append(r[(r.model == model) & (r.metric_name == "rmse")])
        for f in sorted((res / "preds" / f"{args.dataset}_{group}_full").glob(f"{model}_{args.split}_s*_n*.csv")):
            seed, n = f.stem.removeprefix(f"{model}_{args.split}_s").split("_n")
            pr = pd.read_csv(f)
            keep = ~pr.inchikey.isin(excl)
            for col in [c for c in pr.columns if c.startswith("pred_at_")]:
                pred_rows.append({"model": model, "split_strategy": args.split, "split_seed": int(seed), "n_labels": n,
                                  "hp_n_trials": int(col.removeprefix("pred_at_")), "n_excluded": int((~keep).sum()),
                                  "rmse_preds": rmse(pr.y_true, pr[col]),
                                  "rmse_excl": rmse(pr.y_true[keep], pr[col][keep]),
                                  "worst_excluded": float((pr[col] - pr.y_true)[~keep].abs().max())
                                  if (~keep).any() else np.nan})
    main_rmse, rerun_rmse = pd.concat(main_rmse), pd.concat(rerun_rmse)
    pr = pd.DataFrame(pred_rows)

    chk = rerun_rmse[KEY + ["metric_value"]].merge(main_rmse[KEY + ["metric_value"]], on=KEY, suffixes=("_rerun", "_main"))
    chk = chk.merge(pr[KEY + ["rmse_preds"]], on=KEY)
    rerun_diff = chk.groupby("model").apply(lambda g: float((g.metric_value_rerun - g.metric_value_main).abs().max()),
                                            include_groups=False)
    pred_diff = float((chk.rmse_preds - chk.metric_value_rerun).abs().max())
    if len(chk) != len(rerun_rmse) or (rerun_diff > 1e-9).any():
        raise SystemExit(f"재실행이 주 그리드와 다르다 (짝 {len(chk)}/{len(rerun_rmse)}, 최대 차이 {rerun_diff.to_dict()})")

    hit = pr[pr.n_excluded > 0]
    md = [f"# 사후 민감도: test에서 분자 제외 — {args.dataset}", "",
          f"- 제외 분자: {', '.join(f'`{s}`' for s in args.exclude_smiles)} · 분할 {args.split} · 주 지표 RMSE",
          "- **사후 분석이다.** 결과를 본 뒤 정한 분석이므로 주 결과(`reports/gnn_aqsoldb.md`)를 대체하지 않는다",
          f"- 분자가 test에 든 시드: {sorted(hit.split_seed.unique().tolist())} (그 외 시드는 값이 그대로다)",
          f"- 재실행 = 주 그리드: 짝 {len(chk)}개, 모델별 최대 |Δ| "
          + ", ".join(f"{k} {v:.1e}" for k, v in rerun_diff.items())
          + f" · 저장 예측으로 다시 계산한 RMSE와 CSV 최대 |Δ| {pred_diff:.1e}",
          "- HP 선택은 그대로 둔다. 분자가 valid에 든 시드(선택에 영향)는 이 분석으로 보정되지 않는다", "",
          "## 1. 분자 포함 / 제외 RMSE (해당 시드)", "",
          "| 모델 | 시드 | 라벨 수 | 탐색 | 포함 | 제외 | 차이 | 그 분자 절대오차 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    hit = hit.sort_values(["split_seed", "n_labels", "model", "hp_n_trials"],
                          key=lambda s: s.map(budget_order) if s.name == "n_labels" else s)
    md += [f"| {x.model} | {x.split_seed} | {x.n_labels} | {x.hp_n_trials} | {fmt(x.rmse_preds)} | {fmt(x.rmse_excl)} | "
           f"{fmt(x.rmse_excl - x.rmse_preds)} | {fmt(x.worst_excluded, 2)} |" for x in hit.itertuples()]
    md.append("")

    d = main_rmse[main_rmse.hp_n_trials == args.n_trials][KEY + ["metric_value"]]
    d = d.merge(hit[KEY + ["rmse_excl"]], on=KEY, how="left")
    d["metric_excl"] = d.rmse_excl.fillna(d.metric_value)
    md += [f"## 2. C − B 시드 쌍 Wilcoxon, 분자 포함 vs 제외 (탐색 {args.n_trials}회, 예산 간 Holm)", "",
           "| 라벨 수 | 포함: 중앙값 차이 | 포함: p (Holm) | 포함: B 승 | 제외: 중앙값 차이 | 제외: p (Holm) | 제외: B 승 |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    out = {}
    for col in ("metric_value", "metric_excl"):
        block = []
        for n in sorted(d.n_labels.unique(), key=budget_order):
            x = d[d.n_labels == n].pivot_table(index="split_seed", columns="model", values=col)
            w = paired_wilcoxon(x["C_dmpnn"], x["B_desc_lgbm"])
            block.append({"n": n, **w, "b_wins": int((x["C_dmpnn"] > x["B_desc_lgbm"]).sum()), "pairs": len(x)})
        for w, ph in zip(block, holm([w["p_value"] for w in block])):
            w["p_holm"] = ph
        out[col] = block
    for a, b in zip(out["metric_value"], out["metric_excl"]):
        md.append(f"| {a['n']} | {fmt(a['median_diff'])} | {fmt(a['p_holm'], 4)} | {a['b_wins']}/{a['pairs']} | "
                  f"{fmt(b['median_diff'])} | {fmt(b['p_holm'], 4)} | {b['b_wins']}/{b['pairs']} |")
    md.append("")

    path = ROOT / "reports" / f"sensitivity_exclusion_{args.dataset}.md"
    path.write_text("\n".join(md))
    print(f"→ {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
