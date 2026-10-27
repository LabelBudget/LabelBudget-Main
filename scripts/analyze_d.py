"""D(사전학습) 분석: 사전 등록 판정 규칙(05 §4)대로 D − C_fixed, D − C v2, D − B를 검정한다.

python scripts/analyze_d.py --dataset aqsoldb
python scripts/analyze_d.py --dataset aqsoldb --split random       # H3 (05 §2.2 U1)
→ reports/d_<dataset>[_<split>].md, reports/figures/<dataset>_budget_curve_d[_<split>].png
  scaffold_random이면 reports/results/<dataset>_best_d.json (U1 규칙으로 고른 D. test를 쓰지 않는다)

- 탐색 횟수는 예산마다 다르다 (05 §2.2 T1): ≤1,000은 20회, ≥3,000은 10회 체크포인트. 모든 모델이 같은 체크포인트를 쓴다
- 결과 파일 이름은 scripts/queue_d.sh 규약: <dataset>_gnn_<tag>[_<split>]_full.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_baselines import LABELS, PRIMARY, SPLIT_LABEL, budget_order, fmt, plot_curves  # noqa: E402
from analyze_gnn import check_alignment  # noqa: E402

from labelbudget.evaluation.stats import holm, paired_wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"
TAGS = {"C_dmpnn_v2": "v2", "C_fixed": "fixed", "D_mask": "dmask", "D_contrast": "dcontrast"}
# 그림 순서: 인접 쌍 색 구분 검증(dataviz validate_palette, light)을 통과한 순서
ORDER = ["B_desc_lgbm", "D_contrast", "C_dmpnn_v2", "D_mask", "C_fixed"]
D_MODELS = ["D_mask", "D_contrast"]


def trials_for(n_labels: str, small: int = 20, large: int = 10, large_from: int = 3000) -> int:
    return large if budget_order(n_labels) >= large_from else small


def at_checkpoint(d: pd.DataFrame, **kw) -> pd.DataFrame:
    """예산별 체크포인트 행만 남긴다 (T1)."""
    want = d.n_labels.map(lambda n: trials_for(n, **kw))
    return d[d.hp_n_trials == want]


def load(dataset: str, split: str) -> tuple[pd.DataFrame, list[str]]:
    base = pd.read_csv(RES / f"{dataset}_baselines_full.csv", dtype={"n_labels": str})
    base = base[(base.split_strategy == split) & (base.model == "B_desc_lgbm")]
    frames, found = [base], ["B_desc_lgbm"]
    suffix = "" if split == "scaffold_random" else f"_{split}"
    for m, tag in TAGS.items():
        f = RES / f"{dataset}_gnn_{tag}{suffix}_full.csv"
        if not f.exists():
            continue
        g = pd.read_csv(f, dtype={"n_labels": str})
        g = g[(g.model == m) & (g.split_strategy == split)]
        if g.empty:
            continue
        check_alignment(g, base, "B_desc_lgbm")
        frames.append(g)
        found.append(m)
    return pd.concat(frames), found


def compare(w: pd.DataFrame, a: str, b: str, higher: bool) -> list[dict]:
    """a − b 시드 쌍 Wilcoxon, 예산 간 Holm. 'a_better'는 a가 이긴 시드 수."""
    budgets = sorted(w.index.get_level_values(0).unique(), key=budget_order)
    rows = []
    for n in budgets:
        x = w.loc[n][[a, b]].dropna()
        r = paired_wilcoxon(x[a], x[b])
        diff = x[a] - x[b]
        rows.append({"n_labels": n, "n": len(x), **r,
                     "a_better": int((diff > 0).sum() if higher else (diff < 0).sum())})
    for r, ph in zip(rows, holm([r["p_value"] for r in rows])):
        r["p_holm"] = ph
    return rows


def gain(median_diff: float, higher: bool) -> float:
    """'좋아진 양'으로 부호를 맞춘다 (회귀 RMSE는 작을수록 좋다)."""
    return median_diff if higher else -median_diff


def best_d(d: pd.DataFrame, metric: str, higher: bool) -> dict:
    """05 §2.2 U1: valid 점수만으로 D 1개를 고른다. 예산별 시드 평균 차이 → 예산 평균. 같으면 D_mask."""
    v = d[d.model.isin(D_MODELS) & (d.metric_name == metric)]
    w = v.pivot_table(index=["n_labels", "split_seed"], columns="model", values="hp_valid_score")
    per_budget = (w["D_mask"] - w["D_contrast"]).groupby(level=0).mean()
    score = float(per_budget.mean())  # 양수면 D_mask의 valid 점수가 크다
    mask_better = score > 0 if higher else score < 0
    pick = "D_mask" if (mask_better or score == 0) else "D_contrast"
    return {"best_d": pick, "rule": "05 §2.2 U1 (valid only)", "mean_valid_diff_mask_minus_contrast": score,
            "per_budget": {k: float(x) for k, x in per_budget.items()}}


def h2_text(rows: list[dict], higher: bool, who: str = "D") -> str:
    """05 §4 H2: who 우세로 Holm 유의한 가장 큰 예산, B 우세로 유의한 가장 작은 예산."""
    a_win = [x["n_labels"] for x in rows if x["p_holm"] < 0.05 and gain(x["median_diff"], higher) > 0]
    b_win = [x["n_labels"] for x in rows if x["p_holm"] < 0.05 and gain(x["median_diff"], higher) < 0]
    if not a_win and not b_win:
        return f"교차점 없음 ({who}·B 어느 쪽도 Holm 유의한 예산 없음)"
    return (f"{who} 우세로 유의한 가장 큰 예산 {max(a_win, key=budget_order) if a_win else '없음'}, "
            f"B 우세로 유의한 가장 작은 예산 {min(b_win, key=budget_order) if b_win else '없음'}")


def h3_section(dataset: str, split: str, w: pd.DataFrame, models: list[str], metric: str, higher: bool) -> list[str]:
    """05 §2.2 U1: H2 규칙을 이 분할과 scaffold_random에 똑같이 적용해 나란히 적는다 (검정하지 않음)."""
    base, _ = load(dataset, "scaffold_random")
    b = at_checkpoint(base)
    w0 = b[b.metric_name == metric].pivot_table(index=["n_labels", "split_seed"], columns="model", values="metric_value")
    md = ["## 4. H3: 교차점이 분할에 따라 이동하는가 (05 §2.2 U1, 기술만)", "",
          f"H2 규칙을 scaffold_random과 {SPLIT_LABEL.get(split, split)}에 똑같이 적용했다. "
          "가설은 \"교차점이 random보다 scaffold에서 오른쪽\"이다.", "",
          f"| 모델 − B | scaffold_random | {split} |", "|---|---|---|"]
    for m in [m for m in models if m != "B_desc_lgbm" and m in w0]:
        who = "D" if m in D_MODELS else "C"
        md.append(f"| {LABELS[m]} | {h2_text(compare(w0, m, 'B_desc_lgbm', higher), higher, who)} | "
                  f"{h2_text(compare(w, m, 'B_desc_lgbm', higher), higher, who)} |")
    return md + [""]


def pretrain_summary() -> list[str]:
    md = ["## 사전학습 기록", "", "| 방법 | 가중치 SHA-256 | 에폭 | 마지막 train 손실 | 마지막 valid 손실 | valid 정확도 | 총 시간 (h) |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for m in D_MODELS:
        f = ROOT / "models" / "pretrained" / f"{m}.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text())
        h = j["history"][-1]
        md.append(f"| {m} | `{j['weights_sha256'][:16]}…` | {len(j['history'])} | {h['train_loss']:.4f} | "
                  f"{h['valid_loss']:.4f} | {h['valid_acc']:.3f} | {sum(r['sec'] for r in j['history']) / 3600:.1f} |")
    md += ["", "valid 정확도: 마스킹은 가린 원자의 종류 정답률, 대조학습은 배치 안에서 짝 뷰를 가장 가깝게 찾은 비율.", ""]
    return md


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="scaffold_random")
    p.add_argument("--runtime-exclude", nargs="+", default=[], metavar="MODEL:SEED:BUDGET",
                   help="실행 시간 표에서만 뺄 조합 (CPU 경합). 성능 분석은 그대로")
    args = p.parse_args()

    allk, models = load(args.dataset, args.split)
    task = allk.task.iloc[0]
    metric, higher = PRIMARY[task]
    d = at_checkpoint(allk)
    r = d[d.metric_name == metric]
    counts = r.groupby(["model", "n_labels"]).split_seed.nunique()
    w = r.pivot_table(index=["n_labels", "split_seed"], columns="model", values="metric_value")
    budgets = sorted(w.index.get_level_values(0).unique(), key=budget_order)
    tag = "" if args.split == "scaffold_random" else f"_{args.split}"
    better = "클수록" if higher else "작을수록"

    t = r.groupby(["split_strategy", "model", "n_labels"]).metric_value.agg(["mean", "std", "count"]).reset_index()
    t["n_train"] = r.groupby(["split_strategy", "model", "n_labels"]).n_train.mean().to_numpy()
    fig = ROOT / "reports" / "figures" / f"{args.dataset}_budget_curve_d{tag}.png"
    plot_curves(t, metric, fig, f"{args.dataset} — 사전학습 D vs C·B, test {metric.upper()} "
                                f"(탐색 ≤1,000개 20회 · ≥3,000개 10회)", order=ORDER)

    md = [f"# D(사전학습) 결과 — {args.dataset} · {SPLIT_LABEL.get(args.split, args.split)}", "",
          f"- 판정 규칙: `docs/design/05_PRETRAIN_SPEC.md` §4 (결과 전 고정, 커밋 3f50a0f). 주 지표 **{metric.upper()}** ({better} 좋음)",
          "- 탐색 횟수: 라벨 ≤1,000개는 20회, ≥3,000개는 10회 체크포인트. 모든 모델이 같은 체크포인트 (05 §2.2 T1)",
          f"- 포함 모델: {', '.join(LABELS[m] for m in models)}. 모든 GNN 행이 같은 (시드, 예산, 체크포인트)의 B 행과 크기 일치",
          "- 미세조정: CPU 결정론, 프로세스 1스레드. 사전학습 가중치 SHA-256은 각 행의 config_hash에 들어 있다", "",
          f"![예산 곡선](figures/{fig.name})", "", "## 1. 예산별 성능 (평균 ± SD, 시드 수)", "",
          "| 라벨 수 | " + " | ".join(LABELS[m] for m in models) + " |", "|---:|" + "---:|" * len(models)]
    for n in budgets:
        cells = []
        for m in models:
            x = w.loc[n][m].dropna() if m in w else pd.Series(dtype=float)
            cells.append(f"{fmt(x.mean())} ± {fmt(x.std())} ({len(x)})" if len(x) else "—")
        md.append(f"| {n} | " + " | ".join(cells) + " |")
    incomplete = counts[counts < 10]
    if len(incomplete):
        md += ["", "**시드 10개가 다 차지 않은 칸** (05 §2.2: 부분 분석하지 않고 미완료로 보고): "
               + ", ".join(f"{m} {n} ({k}개)" for (m, n), k in incomplete.items())]
    md.append("")

    ds = [m for m in D_MODELS if m in models]
    refs = [x for x in ("C_fixed", "C_dmpnn_v2", "B_desc_lgbm") if x in models]
    md += ["## 2. 사전 등록 비교 (시드 쌍 Wilcoxon, 비교마다 예산 간 Holm)", "",
           f"차이 = D − 비교 대상의 {metric.upper()}. {'양수' if higher else '음수'}면 D가 낫다. 'D 승'은 D가 이긴 시드 수.", ""]
    results = {}
    for dm in ds:
        for ref in refs:
            rows = compare(w, dm, ref, higher)
            results[(dm, ref)] = rows
            md += [f"### {LABELS[dm]} − {LABELS[ref]}", "", "| 라벨 수 | 쌍 | 중앙값 차이 | D 승 | p | p (Holm) |",
                   "|---:|---:|---:|---:|---:|---:|"]
            md += [f"| {x['n_labels']} | {x['n']} | {fmt(x['median_diff'])} | {x['a_better']}/{x['n']} | "
                   f"{fmt(x['p_value'], 4)} | {fmt(x['p_holm'], 4)} |" for x in rows]
            md.append("")

    md += ["## 3. 가설 판정 (05 §4 규칙 그대로)", ""]
    for dm in ds:
        md += [f"### {LABELS[dm]}", ""]
        if (dm, "C_fixed") in results:
            g = [gain(x["median_diff"], higher) for x in results[(dm, "C_fixed")]]
            mono = all(a >= b for a, b in zip(g, g[1:]))
            md.append(f"- **H1** (D − C_fixed 이득이 예산이 늘수록 줄어드는가, 기술만): 이득 중앙값 "
                      + " → ".join(f"{n} {v:+.3f}" for n, v in zip(budgets, g))
                      + f" — {'단조 감소' if mono else '단조 감소 아님'}")
        if (dm, "B_desc_lgbm") in results:
            md.append(f"- **H2** (교차점): {h2_text(results[(dm, 'B_desc_lgbm')], higher)}")
        if (dm, "C_fixed") in results and (dm, "C_dmpnn_v2") in results:
            md.append("- **H4** (구조 탐색 효과와 함께 보기): 예산별 이득 중앙값 D − C_fixed / D − C v2 = "
                      + ", ".join(f"{a['n_labels']} {gain(a['median_diff'], higher):+.3f} / {gain(b['median_diff'], higher):+.3f}"
                                  for a, b in zip(results[(dm, 'C_fixed')], results[(dm, 'C_dmpnn_v2')])))
        md.append("")
    if "C_fixed" in models and "C_dmpnn_v2" in models:
        rows = compare(w, "C_dmpnn_v2", "C_fixed", higher)
        md += ["### 구조 탐색 효과: C v2 − C_fixed (같은 공간에서 d_h·depth를 탐색한 효과)", "",
               "| 라벨 수 | 중앙값 차이 | C v2 승 | p (Holm) |", "|---:|---:|---:|---:|"]
        md += [f"| {x['n_labels']} | {fmt(x['median_diff'])} | {x['a_better']}/{x['n']} | {fmt(x['p_holm'], 4)} |"
               for x in rows]
        md.append("")

    if args.split != "scaffold_random":
        md += h3_section(args.dataset, args.split, w, models, metric, higher)

    if args.split == "scaffold_random" and all(m in models for m in D_MODELS):
        pick = best_d(d, metric, higher)
        (RES / f"{args.dataset}_best_d.json").write_text(json.dumps(pick, indent=1, ensure_ascii=False))
        md += ["## 4. H3·불확실성·앱에 쓸 D (05 §2.2 U1, valid만 사용)", "",
               f"예산별 시드 평균 valid 점수 차이(D_mask − D_contrast): "
               + ", ".join(f"{k} {v:+.4f}" for k, v in sorted(pick["per_budget"].items(), key=lambda kv: budget_order(kv[0])))
               + f" → 예산 평균 {pick['mean_valid_diff_mask_minus_contrast']:+.4f} → **{pick['best_d']}**", ""]

    sup = allk[(allk.metric_name == metric) & (allk.hp_n_trials == 20)
               & allk.n_labels.map(lambda n: budget_order(n) >= 3000) & allk.model.isin(["B_desc_lgbm", "C_dmpnn_v2"])]
    if sup.model.nunique() == 2:
        sw = sup.pivot_table(index=["n_labels", "split_seed"], columns="model", values="metric_value")
        rows10 = compare(w[["C_dmpnn_v2", "B_desc_lgbm"]].loc[[n for n in budgets if budget_order(n) >= 3000]],
                         "C_dmpnn_v2", "B_desc_lgbm", higher)
        rows20 = compare(sw, "C_dmpnn_v2", "B_desc_lgbm", higher)
        md += ["## 5. 보조: 큰 예산의 탐색 10회 vs 20회 (C v2 − B)", "",
               "T1 민감도. 20회 값은 이전 보고서(gnn_v2_*)와 같다. Holm은 두 예산 안에서만 보정한 값이다.", "",
               "| 라벨 수 | 10회 중앙값 차이 | p (Holm) | 20회 중앙값 차이 | p (Holm) |", "|---:|---:|---:|---:|---:|"]
        md += [f"| {a['n_labels']} | {fmt(a['median_diff'])} | {fmt(a['p_holm'], 4)} | {fmt(b['median_diff'])} | "
               f"{fmt(b['p_holm'], 4)} |" for a, b in zip(rows10, rows20)]
        md.append("")

    md += pretrain_summary()

    rt = d[(d.metric_name == metric) & (d.model != "B_desc_lgbm")]
    key = rt.model + ":" + rt.split_seed.astype(str) + ":" + rt.n_labels
    excluded = int(key.isin(args.runtime_exclude).sum())
    rt = rt[~key.isin(args.runtime_exclude)]
    tab = rt.groupby(["model", "n_labels"]).runtime_sec.agg(["count", "mean"]).reset_index()
    md += ["## 실행 시간 (조합당 분, 1스레드 프로세스 4~6개 동시 실행)", "",
           "다른 앱의 CPU 경합이 있던 시간대 값이 섞여 있다. 비용 추정에는 경합이 없던 조합만 쓴다."
           + (f" 경합 구간 {excluded}개 조합은 이 표에서 뺐다." if excluded else ""), "",
           "| 모델 | " + " | ".join(budgets) + " |", "|---|" + "---:|" * len(budgets)]
    for m in [m for m in ORDER if m in set(tab.model)]:
        x = tab[tab.model == m].set_index("n_labels")
        md.append(f"| {LABELS[m]} | " + " | ".join(f"{x.loc[n, 'mean'] / 60:.1f}" if n in x.index else "—"
                                                   for n in budgets) + " |")
    md.append("")

    out = ROOT / "reports" / f"d_{args.dataset}{tag}.md"
    out.write_text("\n".join(md))
    print(f"→ {out.relative_to(ROOT)}\n→ {fig.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
