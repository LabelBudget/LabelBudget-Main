"""불확실성 비교 (05 §2.2 U2): deep ensemble · MC dropout · split conformal. AqSolDB, scaffold_random.

python scripts/run_uncertainty.py --models C_dmpnn_v2 D_mask --workers 6
→ reports/results/uncertainty/<model>_s<seed>_n<budget>.csv (조합마다 test 분자별 예측·표준편차·구간)

- 새 탐색은 하지 않는다. 주 그리드가 valid로 고른 HP(예산별 체크포인트, T1)를 결과 CSV의 hp_selected에서 읽는다
- 앙상블 멤버 j=0..4의 시드는 subsample_seed + 10000·j. j=0은 주 그리드의 최종 모델과 같은 학습이므로
  test RMSE가 결과 CSV 값과 1e-9 안에서 같아야 한다 (다르면 그 조합을 실패로 남긴다 — 분할·특성 재구성 검증)
- MC dropout: j=0 모델의 dropout만 켜고 30회 예측. 고른 dropout이 0이면 분산이 0이다 (HP는 바꾸지 않는다)
- conformal: j=0 모델의 valid 절대 잔차에서 ⌈(n+1)·0.9⌉번째 값 q → 구간 ŷ ± q
- 조합마다 별도 파일이라 중단 후 같은 명령으로 이어서 돈다. 프로세스마다 1스레드 (configs gnn.threads)
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
N_MEMBERS, MC_PASSES, ALPHA, SEED_STRIDE = 5, 30, 0.1, 10000


def run_one(job: dict) -> tuple[dict, str]:
    import torch

    from labelbudget.evaluation.budget import subsample_train
    from labelbudget.evaluation.metrics import regression_metrics
    from labelbudget.evaluation.runner import MODELS, PRETRAINED_DIR  # noqa: F401
    from labelbudget.featurize.graph import batch_graphs, mol_graph
    from labelbudget.models.gnn import DMPNNEstimator, _runtime
    from labelbudget.splits.strategies import make_split

    conf, model, seed, n = job["conf"], job["model"], job["seed"], job["n_labels"]
    out = Path(job["out"])
    df = pd.read_csv(ROOT / "data" / "processed" / "aqsoldb.csv")
    y = df.label.to_numpy()
    sp = conf["splits"]
    split = make_split(df, strategy="scaffold_random", seed=seed, frac_train=sp["frac_train"],
                       frac_valid=sp["frac_valid"], frac_test=sp["frac_test"])
    sub_seed = seed + conf["seeds"]["subsample_seed_offset"]
    tr = split.train[subsample_train(y[split.train], n_labels=None if n == "full" else int(n), seed=sub_seed,
                                     task="regression")]
    va, te = split.valid, split.test
    G = {i: mol_graph(df.smiles[i]) for i in np.concatenate([tr, va, te])}
    X = {k: [G[i] for i in idx] for k, idx in (("tr", tr), ("va", va), ("te", te))}
    spec = MODELS[model]
    params = {**spec.fixed, **json.loads(job["hp_selected"])}
    init = ROOT / spec.pretrained if spec.pretrained else None
    gnn = conf["gnn"]

    preds, vpreds = [], []
    for j in range(N_MEMBERS):
        est = DMPNNEstimator("regression", params, sub_seed + SEED_STRIDE * j, valid=(X["va"], y[va]),
                             init_encoder=init, **gnn).fit(X["tr"], y[tr])
        preds.append(est.predict(X["te"]))
        if j == 0:
            vpreds.append(est.predict(X["va"]))
            rmse0 = regression_metrics(y[te], preds[0])["rmse"]
            if abs(rmse0 - job["grid_rmse"]) > 1e-9:
                return job, f"재현 실패: j=0 RMSE {rmse0:.6f} vs 그리드 {job['grid_rmse']:.6f}"
            # MC dropout: dropout 층만 학습 모드로
            m = est.model
            m.eval()
            for mod in m.modules():
                if isinstance(mod, torch.nn.Dropout):
                    mod.train()
            torch.manual_seed(sub_seed)
            mc = []
            with torch.no_grad(), _runtime(True, gnn.get("threads")):
                bg = batch_graphs(X["te"])
                for _ in range(MC_PASSES):
                    mc.append(m(bg).numpy() * est.sigma + est.mu)
            mc = np.stack(mc)
    P = np.stack(preds)
    resid = np.sort(np.abs(y[va] - vpreds[0]))
    k = int(np.ceil((len(resid) + 1) * (1 - ALPHA)))
    q = float(resid[min(k, len(resid)) - 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"inchikey": df.inchikey.to_numpy()[te], "y_true": y[te],
                  **{f"pred_m{j}": P[j] for j in range(N_MEMBERS)},
                  "ens_mean": P.mean(0), "ens_std": P.std(0, ddof=1),
                  "mc_mean": mc.mean(0), "mc_std": mc.std(0, ddof=1), "conformal_q": q,
                  "dropout": params["dropout"]}).to_csv(out.with_suffix(".tmp"), index=False)
    out.with_suffix(".tmp").replace(out)
    return job, "OK"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    args = p.parse_args()
    from analyze_d import TAGS, at_checkpoint

    conf = yaml.safe_load(args.config.read_text())
    out_dir = ROOT / "reports" / "results" / "uncertainty"
    jobs = []
    for model in args.models:
        g = pd.read_csv(ROOT / "reports" / "results" / f"aqsoldb_gnn_{TAGS[model]}_full.csv", dtype={"n_labels": str})
        g = at_checkpoint(g[(g.model == model) & (g.split_strategy == "scaffold_random") & (g.metric_name == "rmse")])
        for x in g.itertuples():
            out = out_dir / f"{model}_s{x.split_seed}_n{x.n_labels}.csv"
            if not out.exists():
                jobs.append({"conf": conf, "model": model, "seed": int(x.split_seed), "n_labels": x.n_labels,
                             "hp_selected": x.hp_selected, "grid_rmse": float(x.metric_value), "out": str(out)})
    jobs.sort(key=lambda j: -(10**9 if j["n_labels"] == "full" else int(j["n_labels"])))
    print(f"{len(jobs)}개 조합, 프로세스 {args.workers}개 → {out_dir}", flush=True)
    failed = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                job, status = f.result()
            except Exception as e:  # noqa: BLE001 — 한 조합의 예외가 나머지를 멈추지 않게
                job, status = {"model": "?", "seed": "?", "n_labels": "?"}, f"예외 {e!r}"
            failed += status != "OK"
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {job['model']} s{job['seed']} n={job['n_labels']} "
                  f"{status}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
