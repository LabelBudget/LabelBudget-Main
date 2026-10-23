"""GNN(C) 그리드 분석: 베이스라인과 같은 분할·시드·예산에서 C vs B·A 비교.

python scripts/analyze_gnn.py --dataset aqsoldb
→ reports/gnn_aqsoldb.md, reports/figures/aqsoldb_budget_curve_gnn.png

베이스라인과 GNN은 libomp 충돌 때문에 다른 프로세스에서 돌았으므로 결과 파일도 다르다.
여기서는 두 파일을 (분할, 시드, 예산) 키로 맞춰 붙이고, 학습·valid·test 크기가 모든 쌍에서 같은지 먼저 확인한다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_baselines import (LABELS, PRIMARY, SPLIT_LABEL, budget_order, fmt, plot_curves,  # noqa: E402
                               summary_table, wilcoxon_table)

from labelbudget.models.gnn_space import GNN_SPACE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEY = ["split_strategy", "split_seed", "n_labels", "hp_n_trials", "metric_name"]


def check_alignment(gnn: pd.DataFrame, base: pd.DataFrame, ref: str) -> int:
    """C 행마다 같은 키의 기준 모델 행이 있고 n_train·n_valid·n_test가 같아야 한다. 맞춘 쌍 수를 돌려준다."""
    sizes = ["n_train", "n_valid", "n_test"]
    j = gnn[KEY + sizes].merge(base[base.model == ref][KEY + sizes], on=KEY, how="left", suffixes=("", "_ref"),
                               indicator=True)
    missing = int((j._merge != "both").sum())
    differ = int((j[sizes].to_numpy() != j[[s + "_ref" for s in sizes]].to_numpy()).any(axis=1).sum())
    if missing or differ:
        raise RuntimeError(f"C와 {ref}의 조합이 맞지 않는다: 짝 없음 {missing}, 크기 불일치 {differ}")
    return len(j)


def boundary_table(gnn: pd.DataFrame) -> pd.DataFrame:
    """선택된 하이퍼파라미터가 탐색 공간 경계에 붙은 비율. 경계에 몰리면 공간이 좁았다는 신호다."""
    rows = []
    for n, g in gnn.groupby("n_labels"):
        hp = pd.DataFrame([json.loads(s) for s in g.hp_selected])
        row = {"n_labels": n, "n": len(hp)}
        for name, (kind, lo, hi) in GNN_SPACE.items():
            if kind == "choice":
                continue
            span = hi - lo if kind in ("int", "float") else np.log(hi) - np.log(lo)
            pos = (hp[name] - lo) / span if kind in ("int", "float") else (np.log(hp[name]) - np.log(lo)) / span
            row[f"{name}_lo"] = float((pos <= 0.05).mean())
            row[f"{name}_hi"] = float((pos >= 0.95).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_labels", key=lambda s: s.map(budget_order))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n-trials", type=int, default=20, help="분석할 탐색 체크포인트")
    p.add_argument("--split", default="scaffold_random")
    args = p.parse_args()

    res_dir = ROOT / "reports" / "results"
    gnn_path, base_path = res_dir / f"{args.dataset}_gnn_full.csv", res_dir / f"{args.dataset}_baselines_full.csv"
    gnn = pd.read_csv(gnn_path, dtype={"n_labels": str})
    base = pd.read_csv(base_path, dtype={"n_labels": str})
    gnn, base = gnn[gnn.split_strategy == args.split], base[base.split_strategy == args.split]
    n_pairs = check_alignment(gnn, base, "B_desc_lgbm")
    task = gnn.task.iloc[0]
    metric, higher = PRIMARY[task]
    allk = pd.concat([base[base.model.isin(["A_ecfp_lgbm", "B_desc_lgbm"])], gnn])
    d = allk[allk.hp_n_trials == args.n_trials]
    t = summary_table(d, metric)
    fig_path = ROOT / "reports" / "figures" / f"{args.dataset}_budget_curve_gnn.png"
    plot_curves(t, metric, fig_path, f"{args.dataset} — D-MPNN(C) vs 베이스라인, test {metric.upper()} "
                                     f"(탐색 {args.n_trials}회)")

    md = [f"# GNN(C) 결과 — {args.dataset}", "",
          f"- 결과 파일: `{gnn_path.relative_to(ROOT)}` + `{base_path.relative_to(ROOT)}` · 분할 **{args.split}** · "
          f"주 지표 **{metric.upper()}** ({'클수록' if higher else '작을수록'} 좋음) · 탐색 {args.n_trials}회 체크포인트",
          f"- 정렬 확인: C 행 {n_pairs}개 모두 같은 (시드, 예산, 체크포인트)의 B 행과 n_train·n_valid·n_test 일치",
          "- C 실행 조건: CPU 결정론 모드, 프로세스 6개 × 1스레드 (결정 G3·G5). 베이스라인은 LightGBM 8스레드", "",
          f"![예산 곡선](figures/{fig_path.name})", "", "## 1. 예산별 성능 (평균 ± SD, 시드 수)", ""]
    piv = t.assign(cell=t["mean"].map(fmt) + " ± " + t["std"].map(fmt) + " (" + t["count"].astype(str) + ")")
    piv = piv.pivot(index="n_labels", columns="model", values="cell")
    piv = piv.reindex(sorted(piv.index, key=budget_order))
    cols = [m for m in LABELS if m in piv.columns]
    md += [f"### {SPLIT_LABEL[args.split]}", "", "| 라벨 수 | " + " | ".join(LABELS[c] for c in cols) + " |",
           "|---:|" + "---:|" * len(cols)]
    md += [f"| {n} | " + " | ".join(piv.loc[n, c] for c in cols) + " |" for n in piv.index]
    md.append("")

    w = wilcoxon_table(d, metric, [("C_dmpnn", "B_desc_lgbm"), ("C_dmpnn", "A_ecfp_lgbm")])
    md += ["## 2. 시드 쌍 Wilcoxon (비교별로 예산 간 Holm 보정)", "",
           f"차이 = 앞 모델 − 뒤 모델의 {metric.upper()}. "
           + ("음수면 앞 모델이 좋다." if not higher else "양수면 앞 모델이 좋다."), "",
           "| 비교 | 라벨 수 | 중앙값 차이 | 쌍 | p | p (Holm) |", "|---|---:|---:|---:|---:|---:|"]
    md += [f"| {r.pair} | {r.n_labels} | {fmt(r.median_diff)} | {r.n_pairs} | {fmt(r.p_value, 4)} | "
           f"{fmt(r.p_holm, 4)} |" for r in w.itertuples()]
    md.append("")

    r = d[d.metric_name == metric].pivot_table(index=["n_labels", "split_seed"], columns="model", values="metric_value")
    diff = (r["C_dmpnn"] - r["B_desc_lgbm"]).unstack("n_labels")
    diff = diff[sorted(diff.columns, key=budget_order)]
    better = (diff > 0) if not higher else (diff < 0)
    md += ["## 3. 시드별 C − B", "", f"양수 = C의 {metric.upper()}가 {'나쁘다' if not higher else '좋다'}. "
           "마지막 줄은 B가 이긴 시드 수.", "",
           "| 시드 | " + " | ".join(diff.columns) + " |", "|---:|" + "---:|" * len(diff.columns)]
    md += [f"| {k} | " + " | ".join(fmt(v) for v in row) + " |" for k, row in diff.iterrows()]
    md += ["| B 승 | " + " | ".join(f"{int(better[c].sum())}/{int(better[c].notna().sum())}" for c in diff.columns)
           + " |", ""]

    c = gnn[gnn.metric_name == metric]
    ck = c.pivot_table(index="n_labels", columns="hp_n_trials", values="metric_value", aggfunc=["mean", "median"])
    ck = ck.reindex(sorted(ck.index, key=budget_order))
    ks = sorted(c.hp_n_trials.unique())
    md += ["## 4. C의 탐색 횟수 체크포인트 효과", "",
           "평균은 시드 하나의 실패에 끌려간다(아래 5회 full 참고). 중앙값을 함께 적는다.", "",
           "| 라벨 수 | " + " | ".join(f"{k}회 평균" for k in ks) + " | " + " | ".join(f"{k}회 중앙값" for k in ks) + " |",
           "|---:|" + "---:|" * (2 * len(ks))]
    md += [f"| {n} | " + " | ".join(fmt(ck.loc[n, ("mean", k)]) for k in ks) + " | "
           + " | ".join(fmt(ck.loc[n, ("median", k)]) for k in ks) + " |" for n in ck.index]
    worst = c.assign(gap=c.metric_value - c.hp_valid_score).nlargest(3, "gap")
    md += ["", "valid와 test 차이가 가장 큰 C 행 (선택은 valid로만 했으므로 test 실패는 그대로 남긴다):", "",
           "| 시드 | 라벨 수 | 탐색 | valid RMSE | test RMSE |", "|---:|---:|---:|---:|---:|"]
    md += [f"| {x.split_seed} | {x.n_labels} | {x.hp_n_trials} | {fmt(x.hp_valid_score)} | {fmt(x.metric_value)} |"
           for x in worst.itertuples()]
    md.append("")

    b = boundary_table(c[c.hp_n_trials == args.n_trials])
    names = [n for n, s in GNN_SPACE.items() if s[0] != "choice"]
    md += [f"## 5. 선택된 하이퍼파라미터의 경계 쏠림 (탐색 {args.n_trials}회)", "",
           "탐색 범위 양 끝 5% 안에 든 비율 (하한 / 상한). 한쪽에 몰리면 범위를 넓혀야 했다는 신호다.", "",
           "| 라벨 수 | 시드 | " + " | ".join(names) + " |", "|---:|---:|" + "---:|" * len(names)]
    md += [f"| {x['n_labels']} | {x['n']} | " + " | ".join(f"{x[f'{m}_lo']:.0%} / {x[f'{m}_hi']:.0%}" for m in names)
           + " |" for _, x in b.iterrows()]
    # 무작위 추출만으로 끝 구간에 들 확률. 정수형은 끝값 하나가 1/(값 개수)라 5%보다 훨씬 크다
    chance = [1 / (GNN_SPACE[m][2] - GNN_SPACE[m][1] + 1) if GNN_SPACE[m][0] == "int" else 0.05 for m in names]
    md.append("| 무작위 기대치 | | " + " | ".join(f"{c:.0%} / {c:.0%}" for c in chance) + " |")
    md.append("")

    rt = c[c.hp_n_trials == args.n_trials].groupby("n_labels").runtime_sec.agg(["mean", "max", "sum"])
    rt = rt.reindex(sorted(rt.index, key=budget_order))
    md += ["## 6. 실행 시간 (C, 조합당: 탐색 20회 + 체크포인트 재학습)", "",
           "1스레드 프로세스 6개가 동시에 돈 값이다. LightGBM(8스레드) 시간과 직접 비교하지 않는다.", "",
           "| 라벨 수 | 평균 (분) | 최대 (분) | 합계 (시간) |", "|---:|---:|---:|---:|"]
    md += [f"| {n} | {x['mean'] / 60:.1f} | {x['max'] / 60:.1f} | {x['sum'] / 3600:.1f} |" for n, x in rt.iterrows()]
    md.append("")

    out = ROOT / "reports" / f"gnn_{args.dataset}.md"
    out.write_text("\n".join(md))
    print(f"→ {out.relative_to(ROOT)}\n→ {fig_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
