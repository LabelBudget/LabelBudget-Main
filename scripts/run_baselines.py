"""베이스라인 A·B 실행. 스펙 02 §7·§11.

python scripts/run_baselines.py --dataset aqsoldb
python scripts/run_baselines.py --dataset tox21_SR-ARE --strategies scaffold_random --seeds 0 1 --n-trials 10
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from labelbudget.evaluation.runner import BASELINES, VALID_POLICIES, FeatureStore, run_grid

ROOT = Path(__file__).resolve().parents[1]


def _budget(v: str):
    return None if v in ("full", "none", "None") else int(v)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="data/processed/<이름>.csv 의 이름 (예: aqsoldb, tox21_SR-ARE)")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument("--models", nargs="+", choices=list(BASELINES), default=list(BASELINES))
    p.add_argument("--strategies", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--budgets", nargs="+", type=_budget, help="정수 또는 full")
    p.add_argument("--n-trials", type=int)
    p.add_argument("--valid-policy", choices=VALID_POLICIES)
    p.add_argument("--sensitivity", action="store_true",
                   help="민감도 분석: configs sensitivity의 valid 정책·예산으로 실행")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    conf = yaml.safe_load(args.config.read_text())
    df = pd.read_csv(ROOT / "data" / "processed" / f"{args.dataset}.csv")
    task = "classification" if set(df.label.unique()) <= {0, 1} else "regression"
    n_trials = args.n_trials or conf["hp_search"]["n_trials"]
    if not n_trials:
        p.error("n_trials 미정: configs/default.yaml hp_search.n_trials 또는 --n-trials 를 지정하세요")
    if args.sensitivity:
        valid_policy = conf["sensitivity"]["valid_policy"]
        args.budgets = args.budgets or conf["sensitivity"]["budgets"][task]
    else:
        valid_policy = args.valid_policy or conf["hp_search"].get("valid_policy", "full")
    out = args.out or ROOT / "reports" / "results" / f"{args.dataset}_baselines_{valid_policy}.csv"

    features = FeatureStore.build(df, cache=ROOT / "data" / "processed" / f"{args.dataset}_descriptors.csv")
    run_grid(
        df,
        dataset=args.dataset.split("_")[0],
        task=task,
        target_task=args.dataset.split("_", 1)[1] if "_" in args.dataset else df.attrs.get("label_name", "label"),
        features=features,
        models=args.models,
        strategies=args.strategies or conf["splits"]["strategies"],
        split_seeds=args.seeds if args.seeds is not None else conf["seeds"]["split_seeds"],
        budgets=args.budgets or conf["budgets"][task],
        n_trials=n_trials,
        out_path=out,
        trial_checkpoints=conf["hp_search"].get("trial_checkpoints"),
        subsample_seed_offset=conf["seeds"]["subsample_seed_offset"],
        valid_policy=valid_policy,
        frac_train=conf["splits"]["frac_train"],
        frac_valid=conf["splits"]["frac_valid"],
        frac_test=conf["splits"]["frac_test"],
        n_jobs=args.n_jobs,
    )
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
