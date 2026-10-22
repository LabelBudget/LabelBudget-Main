"""GNN 학습 비용 벤치마크. 스펙 04 §4 — 결정 G3(장치)·G5(그리드 범위)의 근거.

python scripts/bench_gnn.py --dataset aqsoldb --devices cpu mps --out docs/bench_gnn.md

베이스라인 그리드가 CPU를 쓰는 동안 재면 숫자가 왜곡된다. 그리드 종료 후 실행할 것.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from labelbudget.evaluation.budget import subsample_train
from labelbudget.evaluation.metrics import regression_metrics
from labelbudget.featurize.graph import mol_graph
from labelbudget.models.gnn import DMPNNEstimator
from labelbudget.splits.strategies import make_split

ROOT = Path(__file__).resolve().parents[1]
# 탐색 공간의 중간값 근처 (chemprop 기본값에 가깝게)
DEFAULT = {"d_h": 300, "depth": 3, "dropout": 0.1, "lr": 1e-3, "ffn_hidden": 300, "ffn_layers": 1, "batch_size": 64}


def _budget(v: str):
    return None if v == "full" else int(v)


def _md_table(df: pd.DataFrame) -> list[str]:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))  # noqa: E731
    return ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns),
            *["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.astype(object).where(df.notna(), None).itertuples(index=False)]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="aqsoldb")
    p.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    p.add_argument("--budgets", nargs="+", type=_budget, default=[100, 1000, None])
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--repeat", type=int, default=2, help="같은 시드 반복 횟수 (재현성 확인)")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    conf = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    gnn = conf["gnn"]
    max_epochs = args.max_epochs or gnn["max_epochs"]
    torch.set_num_threads(args.threads)

    df = pd.read_csv(ROOT / "data" / "processed" / f"{args.dataset}.csv")
    graphs = [mol_graph(s) for s in df.smiles]
    y = df.label.to_numpy()
    split = make_split(df, strategy="scaffold_random", seed=0)
    va, te = split.valid, split.test

    rows = []
    for device in args.devices:
        if device == "mps" and not torch.backends.mps.is_available():
            print("MPS 사용 불가 — 건너뜀")
            continue
        for n in args.budgets:
            tr = split.train[subsample_train(y[split.train], n_labels=n, seed=1000, task="regression")]
            preds = []
            for rep in range(args.repeat):
                t0 = time.time()
                est = DMPNNEstimator("regression", DEFAULT, seed=0, valid=([graphs[i] for i in va], y[va]),
                                     max_epochs=max_epochs, patience=gnn["patience"], device=device,
                                     deterministic=device == "cpu")  # MPS는 결정론 모드 불가
                est.fit([graphs[i] for i in tr], y[tr])
                sec = time.time() - t0
                pred = est.predict([graphs[i] for i in te])
                preds.append(pred)
                rows.append({"device": device, "deterministic": est.deterministic, "n_labels": "full" if n is None else n, "rep": rep,
                             "fit_sec": round(sec, 1), "epochs": len(est.history),
                             "sec_per_epoch": round(sec / max(len(est.history), 1), 2),
                             "best_epoch": est.best_epoch, "test_rmse": regression_metrics(y[te], pred)["rmse"]})
                print(json.dumps(rows[-1], ensure_ascii=False))
            if len(preds) > 1:
                rows[-1]["repro_max_abs_diff"] = float(np.abs(preds[0] - preds[-1]).max())

    res = pd.DataFrame(rows)
    md = ["# GNN 벤치마크", "", f"- 데이터: {args.dataset}, scaffold_random 시드 0, 기본 HP `{DEFAULT}`",
          f"- max_epochs {max_epochs}, patience {gnn['patience']}, CPU 스레드 {args.threads}", "",
          *_md_table(res), ""]
    if args.out:
        args.out.write_text("\n".join(md))
        print(f"→ {args.out}")
    else:
        print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
