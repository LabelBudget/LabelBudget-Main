"""Chemprop 재현(G4-b) 분석: 직접 구현 C vs 공식 chemprop, 초기화 잡음 바닥과 비교.

python scripts/analyze_chemprop.py --dataset aqsoldb
→ reports/chemprop_aqsoldb.md, reports/figures/aqsoldb_chemprop_delta.png

판정 규칙 (결과를 보기 전에 정함)
- 구현 검증: matched − C(같은 HP, 다른 구현)의 크기가 reinit − C(같은 HP, 같은 구현, 다른 초기화)와 비슷하고
  matched − C의 부호에 치우침이 없으면(시드 쌍 Wilcoxon, 예산 간 Holm) "구현 차이는 초기화 잡음 안에 있다"
- 실무 기준점: default − C, default − B (시드 쌍 Wilcoxon, 예산 간 Holm)
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_baselines import GRID, INK, INK2, SURFACE, budget_order, fmt  # noqa: E402

from labelbudget.evaluation.stats import holm, paired_wilcoxon  # noqa: E402

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parents[1]
NAMES = {"C_dmpnn": "C 직접 구현 (탐색 20회)", "C_dmpnn_reinit": "C 재초기화 (같은 HP)",
         "C_chemprop_matched": "chemprop matched (같은 HP)", "C_chemprop_default": "chemprop 기본값",
         "B_desc_lgbm": "B 서술자+LGBM"}
# 기준(잡음 바닥)은 무채색, 비교 대상은 범주 슬롯 6·7 (dataviz 기준 팔레트, 검증 통과)
DELTA_COLORS = {"C_dmpnn_reinit": "#8a8984", "C_chemprop_matched": "#008300", "C_chemprop_default": "#6250d6"}


def load(dataset: str, split: str, n_trials: int) -> pd.DataFrame:
    res = ROOT / "reports" / "results"
    gnn = pd.read_csv(res / f"{dataset}_gnn_full.csv", dtype={"n_labels": str})
    base = pd.read_csv(res / f"{dataset}_baselines_full.csv", dtype={"n_labels": str})
    cp = pd.read_csv(res / f"{dataset}_chemprop_full.csv", dtype={"n_labels": str})
    ours = pd.concat([gnn, base[base.model == "B_desc_lgbm"]])
    ours = ours[(ours.hp_n_trials == n_trials) & (ours.split_strategy == split)]
    d = pd.concat([ours, cp[cp.split_strategy == split]])
    return d[d.metric_name == "rmse"].pivot_table(index=["n_labels", "split_seed"], columns="model",
                                                  values="metric_value")


def wilcoxon_by_budget(w: pd.DataFrame, a: str, b: str) -> list[dict]:
    block = []
    for n in sorted(w.index.get_level_values(0).unique(), key=budget_order):
        x = w.loc[n][[a, b]].dropna()
        block.append({"n_labels": n, **paired_wilcoxon(x[a], x[b])})
    for r, p in zip(block, holm([r["p_value"] for r in block])):
        r["p_holm"] = p
    return block


def plot_delta(w: pd.DataFrame, out: Path) -> None:
    budgets = sorted(w.index.get_level_values(0).unique(), key=budget_order)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=INK2, linewidth=1)
    offsets = dict(zip(DELTA_COLORS, (-0.22, 0.0, 0.22)))
    for m, color in DELTA_COLORS.items():
        for i, n in enumerate(budgets):
            dlt = (w.loc[n][m] - w.loc[n]["C_dmpnn"]).dropna().to_numpy()
            x = np.full(len(dlt), i + offsets[m]) + np.linspace(-0.05, 0.05, len(dlt))
            ax.scatter(x, dlt, s=22, color=color, alpha=0.55, edgecolors=SURFACE, linewidths=0.8, zorder=2)
            ax.plot([i + offsets[m] - 0.08, i + offsets[m] + 0.08], [np.median(dlt)] * 2, color=color,
                    linewidth=2.5, solid_capstyle="round", zorder=3)
    ax.set_xticks(range(len(budgets)), [n if n == "full" else f"{int(n):,}" for n in budgets])
    ax.set_xlabel("학습 라벨 수", fontsize=9, color=INK2)
    ax.set_ylabel("Δ test RMSE (모델 − 직접 구현 C)", fontsize=9, color=INK2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    handles = [plt.Line2D([], [], color=c, marker="o", linewidth=2.5) for c in DELTA_COLORS.values()]
    fig.legend(handles, [NAMES[m] for m in DELTA_COLORS], loc="upper center", ncol=3, frameon=False, fontsize=8,
               labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("직접 구현 C 대비 차이 — 점: 시드, 막대: 중앙값. 0보다 위면 C보다 나쁨", fontsize=10, color=INK,
                 y=1.07, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="scaffold_random")
    p.add_argument("--n-trials", type=int, default=20)
    args = p.parse_args()

    w = load(args.dataset, args.split, args.n_trials)
    missing = w.isna().sum()
    if missing.any():
        raise SystemExit(f"빠진 조합이 있다: {missing[missing > 0].to_dict()}")
    budgets = sorted(w.index.get_level_values(0).unique(), key=budget_order)
    cols = [m for m in NAMES if m in w.columns]
    fig = ROOT / "reports" / "figures" / f"{args.dataset}_chemprop_delta.png"
    plot_delta(w, fig)

    md = [f"# Chemprop 재현 (G4-b) — {args.dataset}", "",
          f"- 분할 {args.split}, 시드 {w.index.get_level_values(1).nunique()}개 × 예산 {len(budgets)}개, 주 지표 test RMSE",
          "- 공식 chemprop 2.3.1 CLI (`labelbudget-equiv` 환경)에 우리 분할·서브샘플 인덱스를 그대로 넣었다 "
          "(`scripts/reproduce_chemprop.py`). 예측은 우리 지표 함수로 다시 계산했다",
          "- **matched**: C가 탐색 20회로 고른 HP + C의 학습 규칙 (고정 lr, mean readout, 최대 100에폭, valid RMSE patience 10)",
          "- **재초기화**: 직접 구현 C를 같은 HP로, 초기화·셔플 시드만 바꿔 재학습 → 같은 구현 안의 잡음 바닥",
          "- **chemprop 기본값**: HP 탐색 없음 (hidden 300, depth 3, FFN 300×1, dropout 0, 50에폭, Noam lr 1e-4→1e-3→1e-4, "
          "norm readout, valid loss 최저 체크포인트)", "",
          f"![C 대비 차이](figures/{fig.name})", "", "## 1. 예산별 RMSE (평균 ± SD / 중앙값)", "",
          "| 라벨 수 | " + " | ".join(NAMES[c] for c in cols) + " |", "|---:|" + "---:|" * len(cols)]
    for n in budgets:
        x = w.loc[n]
        md.append(f"| {n} | " + " | ".join(f"{fmt(x[c].mean())} ± {fmt(x[c].std())} / {fmt(x[c].median())}"
                                            for c in cols) + " |")
    md.append("")

    md += ["## 2. 구현 검증: 같은 HP에서 구현을 바꾼 차이 vs 초기화만 바꾼 차이", "",
           "| 라벨 수 | \\|matched − C\\| 중앙값 | \\|재초기화 − C\\| 중앙값 | matched − C 중앙값 | p (Holm) | "
           "재초기화 − C 중앙값 | p (Holm) |", "|---:|---:|---:|---:|---:|---:|---:|"]
    wm, wr = wilcoxon_by_budget(w, "C_chemprop_matched", "C_dmpnn"), wilcoxon_by_budget(w, "C_dmpnn_reinit", "C_dmpnn")
    for n, a, b in zip(budgets, wm, wr):
        x = w.loc[n]
        md.append(f"| {n} | {fmt((x.C_chemprop_matched - x.C_dmpnn).abs().median())} | "
                  f"{fmt((x.C_dmpnn_reinit - x.C_dmpnn).abs().median())} | {fmt(a['median_diff'])} | "
                  f"{fmt(a['p_holm'], 4)} | {fmt(b['median_diff'])} | {fmt(b['p_holm'], 4)} |")
    wmr = wilcoxon_by_budget(w, "C_chemprop_matched", "C_dmpnn_reinit")
    md += ["", "matched − 재초기화 (둘 다 'C와 같은 HP, 다른 난수'. 구현만 다르다):", "",
           "| 라벨 수 | 중앙값 차이 | p (Holm) |", "|---:|---:|---:|"]
    md += [f"| {r['n_labels']} | {fmt(r['median_diff'])} | {fmt(r['p_holm'], 4)} |" for r in wmr]
    md.append("")

    md += ["## 3. 실무 기준점: chemprop 기본값 vs 탐색한 C, vs B", "",
           "| 라벨 수 | 기본값 − C 중앙값 | p (Holm) | 기본값 − B 중앙값 | p (Holm) | B가 이긴 시드 |",
           "|---:|---:|---:|---:|---:|---:|"]
    wd, wb = wilcoxon_by_budget(w, "C_chemprop_default", "C_dmpnn"), wilcoxon_by_budget(w, "C_chemprop_default", "B_desc_lgbm")
    for n, a, b in zip(budgets, wd, wb):
        x = w.loc[n]
        md.append(f"| {n} | {fmt(a['median_diff'])} | {fmt(a['p_holm'], 4)} | {fmt(b['median_diff'])} | "
                  f"{fmt(b['p_holm'], 4)} | {int((x.C_chemprop_default > x.B_desc_lgbm).sum())}/{len(x)} |")
    md.append("")

    cp = pd.read_csv(ROOT / "reports" / "results" / f"{args.dataset}_chemprop_full.csv", dtype={"n_labels": str})
    cp = cp[(cp.metric_name == "rmse") & (cp.split_strategy == args.split)]
    t = cp.groupby(["model", "n_labels"]).agg(epochs=("epochs_run", "median"), minutes=("runtime_sec", "mean")).reset_index()
    md += ["## 4. 학습 길이와 시간 (1스레드, 6프로세스 동시 실행)", "",
           "| 모델 | 라벨 수 | 에폭 중앙값 | 1회 학습 평균 (분) |", "|---|---:|---:|---:|"]
    md += [f"| {NAMES[r.model]} | {r.n_labels} | {r.epochs:.0f} | {r.minutes / 60:.1f} |"
           for r in t.sort_values(["model", "n_labels"], key=lambda s: s.map(budget_order) if s.name == "n_labels" else s
                                  ).itertuples()]
    md.append("")

    out = ROOT / "reports" / f"chemprop_{args.dataset}.md"
    out.write_text("\n".join(md))
    print(f"→ {out.relative_to(ROOT)}\n→ {fig.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
