"""(모델 × 분할 × 예산 × 시드) 순회, tidy 결과 즉시 저장. 스펙 02 §11.

- 분할 시드 k ↔ 서브샘플 시드 k + subsample_seed_offset (§6.1 시드 설계)
- 조합마다 결과를 바로 CSV에 append하고, 이미 있는 조합은 건너뛴다 (resume)
- valid_policy: full = valid 고정 / proportional = valid도 train:valid 비율대로 예산에 맞춰 줄인다
- trial_checkpoints: n_trials회 탐색 한 번으로 앞 k회 최고 모델을 각각 평가해 hp_n_trials=k 행으로 남긴다
- LightGBM(A·B)과 D-MPNN(C)이 같은 탐색·체크포인트 경로를 쓴다 (동일 예산 원칙)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from labelbudget.curate.dedupe import leakage_check
from labelbudget.evaluation.budget import budget_info, subsample_train
from labelbudget.evaluation.metrics import choose_threshold, classification_metrics, regression_metrics
from labelbudget.featurize.descriptors import DescriptorSelector, compute_descriptors
from labelbudget.featurize.fingerprints import ecfp4
from labelbudget.models.gbm import LGBM_SPACE, best_of, make_lgbm, predict, search_trace
from labelbudget.models.gnn_space import GNN_SPACE
from labelbudget.splits.strategies import make_split

if TYPE_CHECKING:
    from labelbudget.featurize.graph import MolGraph


@dataclass(frozen=True)
class ModelSpec:
    features: tuple[str, ...]
    space: dict
    kind: str  # lgbm | gnn


MODELS: dict[str, ModelSpec] = {
    "A_ecfp_lgbm": ModelSpec(("ecfp",), LGBM_SPACE, "lgbm"),
    "B_desc_lgbm": ModelSpec(("desc",), LGBM_SPACE, "lgbm"),
    "AB_ecfp_desc_lgbm": ModelSpec(("ecfp", "desc"), LGBM_SPACE, "lgbm"),
    "C_dmpnn": ModelSpec(("graph",), GNN_SPACE, "gnn"),
}
BASELINES: dict[str, tuple[str, ...]] = {k: v.features for k, v in MODELS.items() if v.kind == "lgbm"}
VALID_POLICIES = ("full", "proportional")
KEY_COLS = ["dataset", "target_task", "model", "split_strategy", "split_seed", "n_labels", "valid_policy"]


def config_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class FeatureStore:
    ecfp: np.ndarray
    desc: pd.DataFrame
    smiles: list[str] = field(default_factory=list)
    _graphs: list[MolGraph] | None = None

    @property
    def graphs(self) -> list[MolGraph]:
        if self._graphs is None:
            from labelbudget.featurize.graph import mol_graph  # torch를 끌어온다 — GNN 경로에서만
            self._graphs = [mol_graph(s) for s in self.smiles]
        return self._graphs

    @classmethod
    def build(cls, df: pd.DataFrame, cache: Path | None = None) -> FeatureStore:
        desc = None
        if cache is not None and cache.exists():
            cached = pd.read_csv(cache)
            if cached["inchikey"].tolist() == df["inchikey"].tolist():
                desc = cached.drop(columns="inchikey")
        if desc is None:
            desc = compute_descriptors(df.smiles.tolist())
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                desc.assign(inchikey=df.inchikey.to_numpy()).to_csv(cache, index=False)
        return cls(ecfp=ecfp4(df.smiles.tolist()), desc=desc, smiles=df.smiles.tolist())

    def matrices(self, kinds: Sequence[str], tr, *others) -> tuple[list[np.ndarray], dict]:
        """학습 인덱스 tr로 서술자 컬럼을 고정하고 [X_tr, *X_others]를 만든다. graph는 MolGraph 목록."""
        if "graph" in kinds:
            return [[self.graphs[i] for i in idx] for idx in (tr, *others)], {"graph": "chemprop_v2_featurizer"}
        parts, info = [[] for _ in range(1 + len(others))], {}
        if "desc" in kinds:
            sel = DescriptorSelector().fit(self.desc.iloc[tr])
            for p, idx in zip(parts, (tr, *others)):
                p.append(sel.transform(self.desc.iloc[idx]))
            info["desc"] = sel.summary()
        if "ecfp" in kinds:
            for p, idx in zip(parts, (tr, *others)):
                p.append(self.ecfp[idx].astype(np.float32))
            info["n_ecfp_bits"] = self.ecfp.shape[1]
        return [np.hstack(p) for p in parts], info


def _done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    prev = pd.read_csv(path, usecols=KEY_COLS, dtype=str)
    return set(map(tuple, prev[KEY_COLS].drop_duplicates().to_numpy()))


def _key(row: dict) -> tuple:
    return tuple(str(row[c]) for c in KEY_COLS)


def _proportional_valid(valid, y_valid, n_labels, frac_train, frac_valid, seed, task):
    if n_labels is None:
        return valid
    k = max(int(round(n_labels * frac_valid / frac_train)), 2)
    sub = subsample_train(y_valid, n_labels=k, seed=seed, task=task)
    return valid[sub]


def run_grid(
    df: pd.DataFrame,
    *,
    dataset: str,
    task: str,
    target_task: str,
    features: FeatureStore,
    models: Sequence[str],
    strategies: Sequence[str],
    split_seeds: Sequence[int],
    budgets: Sequence[int | None],
    n_trials: int,
    out_path: Path,
    trial_checkpoints: Sequence[int] | None = None,
    subsample_seed_offset: int = 1000,
    valid_policy: str = "full",
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    n_jobs: int = 1,
    gnn_opts: dict | None = None,
    hash_extra: dict | None = None,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    if valid_policy not in VALID_POLICIES:
        raise ValueError(f"valid_policy는 {VALID_POLICIES} 중 하나")
    y_all = df.label.to_numpy()
    keys = df.inchikey.to_numpy()
    checkpoints = sorted({k for k in (trial_checkpoints or []) if k < n_trials} | {n_trials})
    gnn_opts = gnn_opts or {}
    if n_jobs > 1 and any(MODELS[m].kind == "lgbm" for m in models) and (
            "torch" in sys.modules or any(MODELS[m].kind == "gnn" for m in models)):
        # torch의 libomp와 LightGBM의 libomp가 충돌해 멀티스레드 학습이 세그폴트로 죽는다
        raise RuntimeError("LightGBM(n_jobs>1)과 torch를 한 프로세스에서 쓸 수 없다: 베이스라인과 GNN을 따로 실행할 것")
    if any(MODELS[m].kind == "gnn" for m in models) and "lightgbm" in sys.modules and "torch" not in sys.modules:
        # 반대 방향도 같다: lightgbm이 먼저 실린 뒤 torch를 실으면 torch 멀티스레드 연산이 죽는다
        raise RuntimeError("lightgbm이 torch보다 먼저 로드됐다: GNN은 lightgbm을 import하지 않은 프로세스에서 실행할 것")

    def chash(spec: ModelSpec) -> str:
        extra = {**(hash_extra or {}), **({"gnn": gnn_opts} if spec.kind == "gnn" else {})}
        return config_hash({"n_trials": n_trials, "checkpoints": checkpoints, "space": spec.space,
                            "frac": [frac_train, frac_valid, frac_test], "offset": subsample_seed_offset, **extra})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path)
    for strategy in strategies:
        for split_seed in split_seeds:
            split = make_split(df, strategy=strategy, seed=split_seed,
                               frac_train=frac_train, frac_valid=frac_valid, frac_test=frac_test)
            sub_seed = split_seed + subsample_seed_offset
            for n_labels in budgets:
                tr = split.train[subsample_train(y_all[split.train], n_labels=n_labels, seed=sub_seed, task=task)]
                va = split.valid if valid_policy == "full" else _proportional_valid(
                    split.valid, y_all[split.valid], n_labels, frac_train, frac_valid, sub_seed + 1, task)
                te = split.test
                lk = leakage_check(keys[tr], keys[te])
                if lk["exact_overlap"]:
                    raise RuntimeError(f"누수: {strategy}/{split_seed}/{n_labels} exact_overlap={lk['exact_overlap']}")
                binfo = budget_info(y_all[split.train], np.searchsorted(split.train, tr), n_labels, task)

                for model in models:
                    base = {"dataset": dataset, "task": task, "target_task": target_task, "model": model,
                            "split_strategy": strategy, "split_seed": split_seed,
                            "n_labels": "full" if n_labels is None else n_labels,
                            "subsample_seed": sub_seed, "valid_policy": valid_policy}
                    if _key(base) in done:
                        continue
                    t0 = time.time()
                    spec = MODELS[model]
                    (X_tr, X_va, X_te), finfo = features.matrices(spec.features, tr, va, te)
                    if spec.kind == "lgbm":
                        def model_fn(params, seed):
                            return make_lgbm(task, params, seed, n_jobs=n_jobs)
                    else:
                        from labelbudget.models.gnn import DMPNNEstimator

                        def model_fn(params, seed, X_va=X_va, y_va=y_all[va]):
                            return DMPNNEstimator(task, params, seed, valid=(X_va, y_va), **gnn_opts)
                    trace = search_trace(model_fn, spec.space, X_tr, y_all[tr], X_va, y_all[va],
                                         n_trials=n_trials, seed=sub_seed, task=task)
                    rows, evaluated = [], {}
                    for k in checkpoints:
                        params, val_score = best_of(trace, k, task)
                        pkey = json.dumps(params, sort_keys=True)
                        if pkey not in evaluated:  # 같은 파라미터가 뽑히면 재학습하지 않는다
                            fitted = model_fn(params, sub_seed).fit(X_tr, y_all[tr])
                            p_te = predict(fitted, X_te, task)
                            if task == "regression":
                                evaluated[pkey] = regression_metrics(y_all[te], p_te)
                            else:
                                thr = choose_threshold(y_all[va], predict(fitted, X_va, task))
                                evaluated[pkey] = classification_metrics(y_all[te], p_te, thr)
                        metrics = evaluated[pkey]
                        common = {
                            **base,
                            "n_train": len(tr), "n_valid": len(va), "n_test": len(te),
                            "n_test_positive": int(y_all[te].sum()) if task == "classification" else None,
                            "n_train_positive": binfo.get("n_train_positive"),
                            "low_positive_flag": binfo.get("low_positive_flag", False),
                            "hp_n_trials": k, "hp_selected": pkey,
                            "hp_valid_score": val_score, "features": json.dumps(finfo, sort_keys=True),
                            "leakage_exact": lk["exact_overlap"], "leakage_skeleton": lk["skeleton_overlap"],
                            "config_hash": chash(spec),
                        }
                        rows += [{**common, "metric_name": m, "metric_value": v} for m, v in metrics.items()
                                 if isinstance(v, (int, float)) and not isinstance(v, bool)]
                    runtime = round(time.time() - t0, 2)
                    pd.DataFrame(rows).assign(runtime_sec=runtime).to_csv(
                        out_path, mode="a", header=not out_path.exists(), index=False)
                    done.add(_key(base))
                    main = metrics.get("rmse", metrics.get("auprc"))
                    log(f"{model:18} {strategy:15} s{split_seed} n={base['n_labels']!s:>5} "
                        f"valid={len(va):4} → {main:.3f} @{checkpoints[-1]}  ({runtime:.1f}s)")
    return pd.read_csv(out_path)
