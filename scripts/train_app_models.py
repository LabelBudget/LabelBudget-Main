"""데모 앱용 모델 학습 (05 §2.2 U3). LightGBM과 torch는 같은 프로세스에 실을 수 없어 모델마다 따로 실행한다.

python scripts/train_app_models.py --model B_desc_lgbm
python scripts/train_app_models.py --model D_mask        # reports/results/aqsoldb_best_d.json 의 D
→ models/app/<model>.* + models/app/train_smiles.txt

AqSolDB scaffold_random 시드 0, 전체 예산. HP는 주 그리드가 valid로 고른 값(전체 예산 체크포인트 10회, T1).
학습 뒤 test RMSE가 주 그리드 값과 같은지 확인한다 (같은 학습이므로 같아야 한다).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_d import TAGS, trials_for  # noqa: E402

from labelbudget.evaluation.budget import subsample_train  # noqa: E402
from labelbudget.evaluation.metrics import regression_metrics  # noqa: E402
from labelbudget.splits.strategies import make_split  # noqa: E402

SEED, BUDGET = 0, "full"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    conf = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    df = pd.read_csv(ROOT / "data" / "processed" / "aqsoldb.csv")
    y = df.label.to_numpy()
    sp = conf["splits"]
    split = make_split(df, strategy="scaffold_random", seed=SEED, frac_train=sp["frac_train"],
                       frac_valid=sp["frac_valid"], frac_test=sp["frac_test"])
    sub_seed = SEED + conf["seeds"]["subsample_seed_offset"]
    tr = split.train[subsample_train(y[split.train], n_labels=None, seed=sub_seed, task="regression")]
    va, te = split.valid, split.test

    res = ROOT / "reports" / "results"
    f = res / ("aqsoldb_baselines_full.csv" if args.model == "B_desc_lgbm" else f"aqsoldb_gnn_{TAGS[args.model]}_full.csv")
    g = pd.read_csv(f, dtype={"n_labels": str})
    row = g[(g.model == args.model) & (g.split_strategy == "scaffold_random") & (g.split_seed == SEED)
            & (g.n_labels == BUDGET) & (g.hp_n_trials == trials_for(BUDGET)) & (g.metric_name == "rmse")].iloc[0]
    params = json.loads(row.hp_selected)
    out = ROOT / "models" / "app"
    out.mkdir(parents=True, exist_ok=True)

    if args.model == "B_desc_lgbm":
        from labelbudget.featurize.descriptors import DescriptorSelector
        from labelbudget.models.gbm import make_lgbm
        desc = pd.read_csv(ROOT / "data" / "processed" / "aqsoldb_descriptors.csv").drop(columns="inchikey")
        sel = DescriptorSelector().fit(desc.iloc[tr])
        model = make_lgbm("regression", params, sub_seed, n_jobs=8).fit(sel.transform(desc.iloc[tr]), y[tr])
        pred = model.predict(sel.transform(desc.iloc[te]))
        model.booster_.save_model(str(out / "B_desc_lgbm.txt"))
        (out / "B_desc_lgbm_columns.json").write_text(json.dumps(sel.columns))
    else:
        import torch

        from labelbudget.evaluation.runner import MODELS
        from labelbudget.featurize.graph import mol_graph
        from labelbudget.models.gnn import DMPNNEstimator
        spec = MODELS[args.model]
        full = {**spec.fixed, **params}
        G = [mol_graph(s) for s in df.smiles]
        est = DMPNNEstimator("regression", full, sub_seed, valid=([G[i] for i in va], y[va]),
                             init_encoder=ROOT / spec.pretrained, **conf["gnn"]).fit([G[i] for i in tr], y[tr])
        pred = est.predict([G[i] for i in te])
        # mu·sigma는 numpy 스칼라라 float으로 바꾼다 (torch.load 기본 weights_only=True가 numpy 객체를 거부한다)
        torch.save({"state_dict": est.model.state_dict(), "params": full, "mu": float(est.mu), "sigma": float(est.sigma)},
                   out / f"{args.model}.pt")
    rmse = regression_metrics(y[te], pred)["rmse"]
    if abs(rmse - row.metric_value) > 1e-9:
        raise RuntimeError(f"주 그리드와 다르다: {rmse:.6f} vs {row.metric_value:.6f}")
    (out / "train_smiles.txt").write_text("\n".join(df.smiles.to_numpy()[tr]))
    meta = out / "meta.json"
    m = json.loads(meta.read_text()) if meta.exists() else {}
    m[args.model] = {"split": "scaffold_random", "seed": SEED, "budget": BUDGET, "n_train": len(tr),
                     "hp": params, "test_rmse": rmse, "source": f.name}
    meta.write_text(json.dumps(m, indent=1, ensure_ascii=False))
    print(f"{args.model}: test RMSE {rmse:.4f} (주 그리드와 같음) → {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
