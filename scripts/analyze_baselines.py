"""그리드 결과 분석: 예산 곡선, 시드 쌍 Wilcoxon, 탐색 횟수 효과, valid 민감도.

python scripts/analyze_baselines.py --dataset aqsoldb
→ reports/baselines_aqsoldb.md, reports/figures/aqsoldb_budget_curve.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "DejaVu Sans"]  # 한글 라벨
plt.rcParams["axes.unicode_minus"] = False
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from labelbudget.evaluation.stats import holm, paired_wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# 모델 ↔ 색 고정 (dataviz 기준 팔레트 슬롯 순서. 엔티티를 따라가고 순위로 바꾸지 않는다)
COLORS = {"A_ecfp_lgbm": "#2a78d6", "B_desc_lgbm": "#eb6834", "AB_ecfp_desc_lgbm": "#1baf7a",
          "C_dmpnn": "#eda100", "D_dmpnn_pretrained": "#e87ba4",
          "C_dmpnn_v2": "#eda100",  # C의 개정판(readout 탐색). 같은 엔티티라 같은 색 — 한 그림에 v1과 같이 그리지 않는다
          # 05 D 비교 (팔레트 6·7번 슬롯). 그리는 순서는 analyze_d.ORDER — 인접 쌍 색 구분 검증을 통과한 순서
          "C_fixed": "#008300", "D_mask": "#e87ba4", "D_contrast": "#6250d6"}
LABELS = {"A_ecfp_lgbm": "A  ECFP4+LGBM", "B_desc_lgbm": "B  서술자+LGBM", "AB_ecfp_desc_lgbm": "A+B",
          "C_dmpnn": "C  D-MPNN", "D_dmpnn_pretrained": "D  D-MPNN+사전학습", "C_dmpnn_v2": "C  D-MPNN v2",
          "C_fixed": "C_fixed  크기 고정", "D_mask": "D  마스킹 사전학습", "D_contrast": "D  대조학습 사전학습"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SPLITS = ["random", "scaffold", "scaffold_random"]
SPLIT_LABEL = {"random": "random (낙관적 상한)", "scaffold": "scaffold (고정 test, 참고점)",
               "scaffold_random": "scaffold_random (주 분석)"}
PRIMARY = {"regression": ("rmse", False), "classification": ("auprc", True)}


def budget_order(v) -> float:
    return np.inf if str(v) == "full" else float(v)


def load(path: Path, n_trials: int) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"n_labels": str})
    return d[d.hp_n_trials == n_trials]


def summary_table(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    r = d[d.metric_name == metric]
    g = r.groupby(["split_strategy", "model", "n_labels"]).metric_value
    t = g.agg(["mean", "std", "count"]).reset_index()
    t["n_train"] = r.groupby(["split_strategy", "model", "n_labels"]).n_train.mean().to_numpy()
    return t.sort_values(["split_strategy", "model", "n_labels"], key=lambda s: s.map(budget_order)
                         if s.name == "n_labels" else s)


def wilcoxon_table(d: pd.DataFrame, metric: str, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    r = d[d.metric_name == metric]
    rows = []
    for split in SPLITS:
        for a, b in pairs:
            block = []
            for n in sorted(r.n_labels.unique(), key=budget_order):
                x = r[(r.split_strategy == split) & (r.n_labels == n)].pivot_table(
                    index="split_seed", columns="model", values="metric_value")
                if a not in x or b not in x:
                    continue
                x = x[[a, b]].dropna()
                if len(x) < 2:
                    continue
                w = paired_wilcoxon(x[a], x[b])
                block.append({"split": split, "pair": f"{a} − {b}", "n_labels": n, **w})
            if block:
                adj = holm([w["p_value"] for w in block])
                for w, p in zip(block, adj):
                    w["p_holm"] = p
                rows += block
    return pd.DataFrame(rows)


def _direct_labels(ax, ends, min_gap_pt: float = 11.0) -> None:
    """선 끝 이름표. 겹치지 않도록 화면 좌표에서 최소 간격만큼 위아래로 민다."""
    if not ends:
        return
    ax.figure.canvas.draw()
    to_px = ax.transData.transform
    pt = ax.figure.dpi / 72
    items = sorted(((to_px((x, y))[1], m, x, y) for m, x, y in ends))
    placed = []
    for py, m, x, y in items:
        if placed and py - placed[-1][0] < min_gap_pt * pt:
            py = placed[-1][0] + min_gap_pt * pt
        placed.append((py, m, x, y))
    for py, m, x, y in placed:
        dy = (py - to_px((x, y))[1]) / pt
        ax.annotate(LABELS[m], (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                    fontsize=8, color=INK2)


def plot_curves(t: pd.DataFrame, metric: str, out: Path, title: str, order: list[str] | None = None) -> None:
    splits = [s for s in SPLITS if s in set(t.split_strategy)]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.6 * len(splits), 4.0), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, split in zip(axes, splits):
        ax.set_facecolor(SURFACE)
        sub = t[t.split_strategy == split]
        ends = []
        for model in [m for m in (order or COLORS) if m in set(sub.model)]:
            s = sub[sub.model == model].sort_values("n_train")
            x, y, sd = s.n_train.to_numpy(), s["mean"].to_numpy(), s["std"].fillna(0).to_numpy()
            ax.fill_between(x, y - sd, y + sd, color=COLORS[model], alpha=0.12, linewidth=0)
            ax.plot(x, y, color=COLORS[model], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, solid_capstyle="round")
            ends.append((model, x[-1], y[-1]))
        _direct_labels(ax, ends)
        ax.set_xscale("log")
        # 로그축 보조 눈금 라벨(2×10², 3×10² …)이 예산 사이에서 겹치므로 실제 학습 크기에만 눈금을 단다
        ticks = sorted({int(round(n)) for n in sub.n_train})
        ax.set_xticks(ticks, [f"{n:,}" for n in ticks])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_title(SPLIT_LABEL[split], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("학습 라벨 수 (로그)", fontsize=9, color=INK2)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.margins(x=0.25)
    axes[0].set_ylabel(f"test {metric.upper()} (평균 ± 1 SD, 시드)", fontsize=9, color=INK2)
    shown = [m for m in (order or COLORS) if m in set(t.model)]
    handles = [plt.Line2D([], [], color=COLORS[m], linewidth=2, marker="o") for m in shown]
    fig.legend(handles, [LABELS[m] for m in shown], loc="upper center",
               ncol=len(handles), frameon=False, fontsize=8, labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, fontsize=11, color=INK, y=1.07, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def classification_extras(d: pd.DataFrame) -> list[str]:
    """분류 전용 (02 §8): test 양성 수와 20개 미만 플래그, AUROC를 양성률(=무작위 AUPRC)과 같은 표에."""
    r = d[d.metric_name == "auroc"]
    md = ["## 5. 분류 보조: 양성 수와 AUROC", "",
          "AUPRC의 무작위 기준선은 test 양성률이다. AUROC는 불균형에서 낙관적이므로 양성률과 함께 읽는다. "
          "test 양성 20개 미만 조합은 신뢰할 수 없다(02 §8).", ""]
    for split in SPLITS:
        s = r[r.split_strategy == split]
        if s.empty:
            continue
        one = s.drop_duplicates(["split_seed", "n_labels"])  # 양성 수는 모델과 무관
        pos = one.groupby("n_labels").agg(
            test_pos_min=("n_test_positive", "min"), test_pos_max=("n_test_positive", "max"),
            train_pos_min=("n_train_positive", "min"), train_pos_max=("n_train_positive", "max"),
            rate=("n_test_positive", "sum"), n_test=("n_test", "sum"), flagged=("low_positive_flag", "sum"))
        auroc = s.pivot_table(index="n_labels", columns="model", values="metric_value", aggfunc="mean")
        cols = [m for m in COLORS if m in auroc.columns]
        md += [f"### {SPLIT_LABEL[split]}", "",
               "| 라벨 수 | test 양성 | train 양성 | 양성 20개 미만 | test 양성률 | "
               + " | ".join(f"AUROC {LABELS[c]}" for c in cols) + " |", "|---:|---:|---:|---:|---:|" + "---:|" * len(cols)]
        for n in sorted(pos.index, key=budget_order):
            x = pos.loc[n]
            md.append(f"| {n} | {x.test_pos_min}~{x.test_pos_max} | {x.train_pos_min}~{x.train_pos_max} | "
                      f"{int(x.flagged)}/{len(one[one.n_labels == n])} | {x.rate / x.n_test:.1%} | "
                      + " | ".join(fmt(auroc.loc[n, c]) for c in cols) + " |")
        md.append("")
    return md


def fmt(v, nd=3):
    return "" if pd.isna(v) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n-trials", type=int, default=20, help="분석할 탐색 체크포인트")
    p.add_argument("--group", default="baselines")
    args = p.parse_args()

    res_dir = ROOT / "reports" / "results"
    main_path = res_dir / f"{args.dataset}_{args.group}_full.csv"
    allk = pd.read_csv(main_path, dtype={"n_labels": str})
    task = allk.task.iloc[0]
    metric, higher = PRIMARY[task]
    d = allk[allk.hp_n_trials == args.n_trials]
    t = summary_table(d, metric)
    fig_path = ROOT / "reports" / "figures" / f"{args.dataset}_budget_curve.png"
    plot_curves(t, metric, fig_path, f"{args.dataset} — 라벨 예산별 test {metric.upper()} (탐색 {args.n_trials}회)")

    md = [f"# 베이스라인 결과 — {args.dataset}", "",
          f"- 결과 파일: `{main_path.relative_to(ROOT)}` · 주 지표 **{metric.upper()}** "
          f"({'클수록' if higher else '작을수록'} 좋음) · 탐색 {args.n_trials}회 체크포인트",
          "- valid 정책 full: 학습 라벨 N개 + 하이퍼파라미터 선택용 valid(~10%)", "",
          f"![예산 곡선](figures/{fig_path.name})", "", "## 1. 예산별 성능 (평균 ± SD, 시드 수)", ""]
    for split in SPLITS:
        s = t[t.split_strategy == split]
        if s.empty:
            continue
        piv = s.assign(cell=s["mean"].map(fmt) + " ± " + s["std"].map(fmt) + " (" + s["count"].astype(str) + ")")
        piv = piv.pivot(index="n_labels", columns="model", values="cell")
        piv = piv.reindex(sorted(piv.index, key=budget_order))
        cols = [m for m in COLORS if m in piv.columns]
        md += [f"### {SPLIT_LABEL[split]}", "", "| 라벨 수 | " + " | ".join(LABELS[c] for c in cols) + " |",
               "|---:|" + "---:|" * len(cols)]
        md += [f"| {n} | " + " | ".join(piv.loc[n, c] for c in cols) + " |" for n in piv.index]
        md.append("")

    w = wilcoxon_table(d, metric, [("B_desc_lgbm", "A_ecfp_lgbm"), ("AB_ecfp_desc_lgbm", "B_desc_lgbm")])
    md += ["## 2. 시드 쌍 Wilcoxon (분할 내 예산 간 Holm 보정)", "",
           f"차이 = 앞 모델 − 뒤 모델의 {metric.upper()}. "
           + ("음수면 앞 모델이 좋다." if not higher else "양수면 앞 모델이 좋다."), "",
           "| 분할 | 비교 | 라벨 수 | 중앙값 차이 | 쌍 | p | p (Holm) |", "|---|---|---:|---:|---:|---:|---:|"]
    md += [f"| {r.split} | {r.pair} | {r.n_labels} | {fmt(r.median_diff)} | {r.n_pairs} | {fmt(r.p_value, 4)} | "
           f"{fmt(r.p_holm, 4)} |" for r in w.itertuples()]
    md.append("")

    r = allk[(allk.metric_name == metric) & (allk.split_strategy == "scaffold_random")]
    ck = r.pivot_table(index=["model", "n_labels"], columns="hp_n_trials", values="metric_value", aggfunc="mean")
    md += ["## 3. 탐색 횟수 체크포인트 효과 (scaffold_random, 평균)", "",
           "| 모델 | 라벨 수 | " + " | ".join(f"{k}회" for k in ck.columns) + " |", "|---|---:|" + "---:|" * len(ck.columns)]
    for (m, n), row in sorted(ck.iterrows(), key=lambda kv: (kv[0][0], budget_order(kv[0][1]))):
        md.append(f"| {LABELS.get(m, m)} | {n} | " + " | ".join(fmt(v) for v in row) + " |")
    md.append("")

    sens_path = res_dir / f"{args.dataset}_{args.group}_proportional.csv"
    if sens_path.exists():
        s = load(sens_path, args.n_trials)
        s = s[s.metric_name == metric]
        f = d[(d.metric_name == metric) & d.n_labels.isin(s.n_labels.unique())]
        key = ["model", "split_strategy", "split_seed", "n_labels"]
        j = f[key + ["metric_value"]].merge(s[key + ["metric_value", "n_valid"]], on=key, suffixes=("_full", "_prop"))
        agg = j.groupby(["split_strategy", "model", "n_labels"]).agg(
            full=("metric_value_full", "mean"), prop=("metric_value_prop", "mean"), n_valid=("n_valid", "first"),
            n=("metric_value_full", "size")).reset_index()
        md += ["## 4. valid 민감도 (full vs proportional, 저예산)", "",
               "| 분할 | 모델 | 라벨 수 | valid(prop) | full | proportional | 차이 | 쌍 |",
               "|---|---|---:|---:|---:|---:|---:|---:|"]
        md += [f"| {a.split_strategy} | {LABELS.get(a.model, a.model)} | {a.n_labels} | {a.n_valid} | {fmt(a.full)} | "
               f"{fmt(a.prop)} | {fmt(a.prop - a.full)} | {a.n} |" for a in agg.itertuples()]
        ws = wilcoxon_table(s.assign(metric_name=metric), metric, [("B_desc_lgbm", "A_ecfp_lgbm")])
        if not ws.empty:
            md += ["", "proportional에서의 B − A Wilcoxon:", "",
                   "| 분할 | 라벨 수 | 중앙값 차이 | p (Holm) |", "|---|---:|---:|---:|"]
            md += [f"| {x.split} | {x.n_labels} | {fmt(x.median_diff)} | {fmt(x.p_holm, 4)} |" for x in ws.itertuples()]
        md.append("")

    if task == "classification":
        md += classification_extras(d)

    out = ROOT / "reports" / f"{args.group}_{args.dataset}.md"
    out.write_text("\n".join(md))
    print(f"→ {out.relative_to(ROOT)}\n→ {fig_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
