"""불확실성 비교 분석 (05 §2.2 U2). 기술 통계만 낸다 (검정 없음).

python scripts/analyze_uncertainty.py → reports/uncertainty_aqsoldb.md

- 90% 구간: ensemble·MC는 평균 ± 1.645·표준편차 (정규 가정), conformal은 j=0 예측 ± q
- 포함률이 0.90에 가까울수록 보정이 맞다. 같은 포함률이면 폭이 좁을수록 좋다
- 순위 상관: 표준편차가 큰 분자일수록 |오차|가 큰가 (Spearman, 조합마다 계산 후 시드 평균)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_baselines import LABELS, budget_order  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
Z90 = 1.6449


def combo_stats(d: pd.DataFrame) -> dict:
    y = d.y_true.to_numpy()
    e0, ee, em = np.abs(y - d.pred_m0), np.abs(y - d.ens_mean), np.abs(y - d.mc_mean)
    out = {"rmse_single": float(np.sqrt((e0 ** 2).mean())), "rmse_ens": float(np.sqrt((ee ** 2).mean())),
           "cov_ens": float((ee <= Z90 * d.ens_std).mean()), "width_ens": float(2 * Z90 * d.ens_std.mean()),
           "cov_conf": float((e0 <= d.conformal_q).mean()), "width_conf": float(2 * d.conformal_q.iloc[0]),
           "rho_ens": float(spearmanr(d.ens_std, ee).statistic), "dropout0": bool(d.dropout.iloc[0] == 0)}
    if d.mc_std.max() > 0:
        out |= {"cov_mc": float((em <= Z90 * d.mc_std).mean()), "width_mc": float(2 * Z90 * d.mc_std.mean()),
                "rho_mc": float(spearmanr(d.mc_std, em).statistic)}
    return out


def main() -> int:
    src = ROOT / "reports" / "results" / "uncertainty"
    rows = []
    for f in sorted(src.glob("*.csv")):
        model, s, n = f.stem.rsplit("_", 2)
        rows.append({"model": model, "seed": int(s[1:]), "n_labels": n[1:], **combo_stats(pd.read_csv(f))})
    r = pd.DataFrame(rows)
    for c in ("cov_mc", "width_mc", "rho_mc"):  # 모든 조합의 dropout이 0이면 MC 열이 없다
        r[c] = r[c] if c in r else np.nan
    md = ["# 불확실성 비교 — AqSolDB · scaffold_random", "",
          "- 설계: `docs/design/05_PRETRAIN_SPEC.md` §2.2 U2 (결과 전 고정). 주 그리드가 valid로 고른 HP를 그대로 쓰고 새로 탐색하지 않았다",
          "- deep ensemble 5개 (j=0은 주 그리드 모델과 같다 — 모든 조합에서 test RMSE 재현 확인), MC dropout 30회, split conformal (valid 잔차)",
          "- 목표 포함률 0.90. 값은 시드 10개 평균. 검정하지 않는다", ""]
    for model in sorted(r.model.unique()):
        x = r[r.model == model].groupby("n_labels").agg(
            n=("seed", "count"), rmse_single=("rmse_single", "mean"), rmse_ens=("rmse_ens", "mean"),
            cov_ens=("cov_ens", "mean"), width_ens=("width_ens", "mean"), cov_mc=("cov_mc", "mean"),
            width_mc=("width_mc", "mean"), cov_conf=("cov_conf", "mean"), width_conf=("width_conf", "mean"),
            rho_ens=("rho_ens", "mean"), rho_mc=("rho_mc", "mean"), dropout0=("dropout0", "sum"))
        x = x.reindex(sorted(x.index, key=budget_order))
        md += [f"## {LABELS.get(model, model)}", "",
               "| 라벨 수 | 시드 | RMSE 단일 | RMSE 앙상블 | 포함률 앙상블 | 폭 | 포함률 MC | 폭 | 포함률 conformal | 폭 | ρ 앙상블 | ρ MC | dropout 0 |",
               "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        f3 = lambda v: "—" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
        md += [f"| {n} | {int(v.n)} | {f3(v.rmse_single)} | {f3(v.rmse_ens)} | {f3(v.cov_ens)} | {f3(v.width_ens)} | "
               f"{f3(v.cov_mc)} | {f3(v.width_mc)} | {f3(v.cov_conf)} | {f3(v.width_conf)} | {f3(v.rho_ens)} | "
               f"{f3(v.rho_mc)} | {int(v.dropout0)}/{int(v.n)} |" for n, v in x.iterrows()]
        md += ["", "MC 값은 고른 dropout이 0이 아닌 조합만의 평균이다 (마지막 열이 dropout 0인 조합 수).", ""]
    out = ROOT / "reports" / "uncertainty_aqsoldb.md"
    out.write_text("\n".join(md))
    print(f"→ {out.relative_to(ROOT)} ({len(r)}개 조합)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
