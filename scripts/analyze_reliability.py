"""라벨 신뢰도 층화 평가 (01 §8, 05 §2.2). AqSolDB test를 Group으로 나눠 RMSE를 낸다. 기술 통계만.

python scripts/analyze_reliability.py --best-d D_mask → reports/reliability_aqsoldb.md

- 층: G1(단일 출처, 교차 검증 안 된 라벨) vs G2~G5(복수 출처). AqSolDB 원본 Group 컬럼
- 분자별 test 예측(--save-preds)이 있는 모델만 쓴다. B는 scaffold_random 주 그리드를 --save-preds로 다시 실행한 파일
  (reports/results/preds/aqsoldb_baselines_rerun). 다시 실행한 지표가 원래 결과와 다르면 중단한다
- 체크포인트는 예산별 (05 §2.2 T1)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_baselines import LABELS, budget_order  # noqa: E402
from analyze_d import TAGS, trials_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"


def check_rerun() -> None:
    """B 재실행 지표 = 원래 그리드 지표 (결정론). 다르면 예측 파일이 그리드 결과를 대표하지 않는다."""
    a = pd.read_csv(RES / "aqsoldb_baselines_full.csv", dtype={"n_labels": str})
    b = pd.read_csv(RES / "aqsoldb_baselines_rerun.csv", dtype={"n_labels": str})
    key = ["model", "split_strategy", "split_seed", "n_labels", "hp_n_trials", "metric_name"]
    j = b.merge(a, on=key, suffixes=("_rerun", ""))
    if len(j) != len(b) or not np.allclose(j.metric_value_rerun, j.metric_value, rtol=0, atol=1e-12):
        raise RuntimeError(f"B 재실행이 원래 결과와 다르다: 짝 {len(j)}/{len(b)}, "
                           f"최대 차이 {np.abs(j.metric_value_rerun - j.metric_value).max()}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--best-d", required=True, help="reports/results/aqsoldb_best_d.json의 값")
    args = p.parse_args()
    check_rerun()
    grp = pd.read_csv(ROOT / "data" / "processed" / "aqsoldb.csv", usecols=["inchikey", "Group"]).set_index("inchikey").Group
    dirs = {"B_desc_lgbm": RES / "preds" / "aqsoldb_baselines_rerun",
            "C_dmpnn_v2": RES / "preds" / "aqsoldb_gnn_v2_full",
            args.best_d: RES / "preds" / f"aqsoldb_gnn_{TAGS[args.best_d]}_full"}
    rows = []
    for model, d in dirs.items():
        for f in sorted(d.glob(f"{model}_scaffold_random_s*_n*.csv")):
            seed, n = f.stem.split("_s")[-1].split("_n")
            x = pd.read_csv(f)
            col = f"pred_at_{trials_for(n)}"
            x["stratum"] = np.where(x.inchikey.map(grp) == "G1", "G1 단일 출처", "G2~G5 복수 출처")
            for s, g in x.groupby("stratum"):
                rows.append({"model": model, "seed": int(seed), "n_labels": n, "stratum": s, "n": len(g),
                             "rmse": float(np.sqrt(((g.y_true - g[col]) ** 2).mean()))})
    r = pd.DataFrame(rows)
    t = r.groupby(["model", "n_labels", "stratum"]).agg(rmse=("rmse", "mean"), n=("n", "mean"),
                                                         seeds=("seed", "nunique")).reset_index()
    budgets = sorted(t.n_labels.unique(), key=budget_order)
    md = ["# 라벨 신뢰도 층화 평가 — AqSolDB · scaffold_random", "",
          "- test 분자를 AqSolDB Group으로 나눴다: G1 = 단일 출처(교차 검증 안 된 라벨), G2~G5 = 복수 출처",
          "- 값은 시드 10개 평균 test RMSE. 층마다 분자 수가 달라(아래 n) 직접 비교는 기술적으로만 한다. 검정하지 않는다",
          "- B 예측은 주 그리드를 `--save-preds`로 다시 실행한 것 (모든 지표가 원래 결과와 같음을 확인)", "",
          "| 모델 | 층 | " + " | ".join(budgets) + " | test 분자 수 (평균) |", "|---|---|" + "---:|" * (len(budgets) + 1)]
    for model in [m for m in dirs if m in set(t.model)]:
        for s in sorted(t.stratum.unique()):
            x = t[(t.model == model) & (t.stratum == s)].set_index("n_labels")
            md.append(f"| {LABELS[model]} | {s} | " + " | ".join(f"{x.loc[n, 'rmse']:.3f}" if n in x.index else "—"
                                                              for n in budgets) + f" | {x.n.mean():.0f} |")
    gap = r.pivot_table(index=["model", "n_labels", "seed"], columns="stratum", values="rmse")
    gap = (gap.iloc[:, 0] - gap.iloc[:, 1]).groupby(level=[0, 1]).agg(["median", lambda v: int((v > 0).sum())])
    md += ["", "### 층 차이: G1 − 복수 출처 RMSE (시드 중앙값, 괄호는 G1이 더 나쁜 시드 수)", "",
           "| 모델 | " + " | ".join(budgets) + " |", "|---|" + "---:|" * len(budgets)]
    for model in [m for m in dirs if m in gap.index.get_level_values(0)]:
        md.append(f"| {LABELS[model]} | " + " | ".join(
            f"{gap.loc[(model, n)].iloc[0]:+.3f} ({int(gap.loc[(model, n)].iloc[1])}/10)" if (model, n) in gap.index else "—"
            for n in budgets) + " |")
    md.append("")
    out = ROOT / "reports" / "reliability_aqsoldb.md"
    out.write_text("\n".join(md))
    print(f"→ {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
