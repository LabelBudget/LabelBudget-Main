"""공식 chemprop을 우리 분할에서 실행해 직접 구현 C와 비교한다. 결정 G4-(b).

환경이 둘이라 세 단계로 나뉜다 (chemprop은 RDKit을 내리므로 labelbudget-equiv에만 있다).

  # 1) labelbudget: 분할·서브샘플을 인덱스 JSON으로, C가 고른 HP를 함께 내보낸다
  python scripts/reproduce_chemprop.py export --dataset aqsoldb
  # 2) labelbudget-equiv: chemprop train CLI 실행 (조합 × 모드, 프로세스 여러 개, 각 1스레드, 이어서 실행 가능)
  python scripts/reproduce_chemprop.py run --dataset aqsoldb --modes default matched --workers 6
  # 3) labelbudget: 초기화 대조군 — 직접 구현 C를 같은 HP·다른 초기화 시드로 재학습
  python scripts/reproduce_chemprop.py control --dataset aqsoldb --workers 6
  # 4) labelbudget: 예측을 모아 우리 지표 함수로 계산 → reports/results/<dataset>_chemprop_full.csv
  python scripts/reproduce_chemprop.py collect --dataset aqsoldb

모드
- default: chemprop 기본 설정 그대로 (실무자가 chemprop을 바로 쓰면 얻는 값. HP 탐색 없음)
- matched: C가 탐색 20회로 고른 HP + C의 학습 규칙(고정 lr, mean readout, 최대 100에폭, valid RMSE patience 10).
  남는 차이는 가중치 초기화·셔플 난수, Lightning 학습 루프뿐이다 → 직접 구현의 학습 과정 검증
- control: 직접 구현 C를 같은 HP로, 초기화·셔플 시드만 바꿔(+CONTROL_SEED_OFFSET) 재학습.
  C ↔ control 차이가 "초기화 잡음 바닥"이다. C ↔ matched 차이가 이보다 크지 않으면 구현 차이라고 말할 근거가 없다
- abl_*: 학습 규칙 원인 진단. chemprop 기본값에서 C의 학습 규칙을 하나씩(또는 전부) 바꾼다. 3000·전체 예산에서
  기본값이 C보다 나았던 원인이 학습 규칙인지 HP 선택인지 가른다 (04 §6.2, scripts/analyze_chemprop_ablation.py)

  python scripts/reproduce_chemprop.py run --dataset aqsoldb --modes abl_constlr abl_mean abl_patience abl_allrules \
      --budgets 3000 full
  python scripts/reproduce_chemprop.py collect --dataset aqsoldb --modes abl_constlr abl_mean abl_patience abl_allrules \
      --out aqsoldb_chemprop_ablation.csv
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
# 진단 모드: 기본값에 C의 학습 규칙을 덧씌운다. 고정 lr은 Noam의 최고값 1e-3에 맞춘다.
# 정지 규칙은 --epochs를 50으로 둔다(Noam 감쇠 길이가 에폭 수에 묶여 있어 100으로 바꾸면 스케줄도 바뀐다)
_CONST_LR = ["--warmup-epochs", "0", "--init-lr", "1e-3", "--max-lr", "1e-3", "--final-lr", "1e-3"]
ABLATIONS = {"abl_constlr": _CONST_LR, "abl_mean": ["--aggregation", "mean"],
             "abl_patience": ["--patience", "10", "--tracking-metric", "rmse"],
             "abl_allrules": [*_CONST_LR, "--aggregation", "mean", "--epochs", "100", "--patience", "10",
                              "--tracking-metric", "rmse"]}
MODES = ("default", "matched", "control", *ABLATIONS)
MODEL_NAME = {"default": "C_chemprop_default", "matched": "C_chemprop_matched", "control": "C_dmpnn_reinit",
              **{m: f"C_chemprop_{m}" for m in ABLATIONS}}
CHEMPROP_MODES = ("default", "matched", *ABLATIONS)
CONTROL_SEED_OFFSET = 10_000


def _work(dataset: str) -> Path:
    return ROOT / "data" / "processed" / "chemprop" / dataset


def _budget_key(b: str) -> int:
    return 10**9 if b == "full" else int(b)


def export(args) -> int:
    import numpy as np

    from labelbudget.evaluation.budget import subsample_train
    from labelbudget.splits.strategies import make_split

    conf = yaml.safe_load(args.config.read_text())
    df = pd.read_csv(ROOT / "data" / "processed" / f"{args.dataset}.csv")
    y = df.label.to_numpy()
    work = _work(args.dataset)
    (work / "splits").mkdir(parents=True, exist_ok=True)
    df[["smiles", "label"]].to_csv(work / "data.csv", index=False)
    gnn = pd.read_csv(ROOT / "reports" / "results" / f"{args.dataset}_gnn_full.csv", dtype={"n_labels": str})
    gnn = gnn[(gnn.metric_name == "rmse") & (gnn.hp_n_trials == args.n_trials)]
    sp_conf, offset = conf["splits"], conf["seeds"]["subsample_seed_offset"]
    budgets = ["full" if b is None else str(b) for b in conf["budgets"]["regression"]]
    n = 0
    for s in args.strategies:
        for k in conf["seeds"]["split_seeds"]:
            split = make_split(df, strategy=s, seed=k, frac_train=sp_conf["frac_train"],
                               frac_valid=sp_conf["frac_valid"], frac_test=sp_conf["frac_test"])
            for b in budgets:
                tr = split.train[subsample_train(y[split.train], n_labels=None if b == "full" else int(b),
                                                 seed=k + offset, task="regression")]
                row = gnn[(gnn.split_strategy == s) & (gnn.split_seed == k) & (gnn.n_labels == b)]
                if len(row) != 1:
                    raise SystemExit(f"C 결과가 없다: {s} s{k} n={b} (탐색 {args.n_trials}회)")
                stem = f"{s}_s{k}_n{b}"
                (work / "splits" / f"{stem}.json").write_text(json.dumps(
                    [{"train": tr.tolist(), "val": split.valid.tolist(), "test": split.test.tolist()}]))
                (work / "splits" / f"{stem}.meta.json").write_text(json.dumps(
                    {"split_strategy": s, "split_seed": k, "n_labels": b, "subsample_seed": k + offset,
                     "n_train": len(tr), "n_valid": len(split.valid), "n_test": len(split.test),
                     "c_params": json.loads(row.hp_selected.iloc[0]), "c_rmse": float(row.metric_value.iloc[0]),
                     "gnn": conf["gnn"]}))
                assert np.intersect1d(tr, split.test).size == 0
                n += 1
    print(f"→ {work} (조합 {n}개)")
    return 0


def chemprop_args(mode: str, meta: dict) -> list[str]:
    if mode == "default":
        return []
    if mode in ABLATIONS:
        return list(ABLATIONS[mode])
    p, g = meta["c_params"], meta["gnn"]
    lr = str(p["lr"])
    return ["--message-hidden-dim", str(p["d_h"]), "--depth", str(p["depth"]), "--dropout", str(p["dropout"]),
            "--ffn-hidden-dim", str(p["ffn_hidden"]), "--ffn-num-layers", str(p["ffn_layers"]),
            "-b", str(p["batch_size"]), "--aggregation", "mean",
            "--warmup-epochs", "0", "--init-lr", lr, "--max-lr", lr, "--final-lr", lr,
            "--epochs", str(g["max_epochs"]), "--patience", str(g["patience"]), "--tracking-metric", "rmse"]


def run(args) -> int:
    work = _work(args.dataset)
    exe = Path(sys.executable).with_name("chemprop")
    if not exe.exists():
        raise SystemExit(f"chemprop 실행 파일이 없다: {exe} — labelbudget-equiv 환경에서 실행할 것")
    metas = [m for m in sorted(work.glob("splits/*.meta.json"))
             if not args.budgets or json.loads(m.read_text())["n_labels"] in args.budgets]
    jobs = [(m, mode) for m in metas for mode in args.modes if mode in CHEMPROP_MODES]
    jobs.sort(key=lambda j: -_budget_key(json.loads(j[0].read_text())["n_labels"]))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def one(job):
        meta_path, mode = job
        meta, stem = json.loads(meta_path.read_text()), meta_path.name.removesuffix(".meta.json")
        out = work / "runs" / mode / stem
        if (out / "model_0" / "test_predictions.csv").exists():
            return job, 0, 0.0, True
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(exe), "train", "-i", str(work / "data.csv"), "-s", "smiles", "--target-columns", "label",
               "-t", "regression", "--splits-file", str(meta_path.with_name(f"{stem}.json")),
               "--pytorch-seed", str(meta["subsample_seed"]), "--accelerator", "cpu", "--devices", "1", "-n", "0",
               "--metrics", "rmse", "mae", "--remove-checkpoints", "-o", str(out), *chemprop_args(mode, meta)]
        t0 = time.time()
        with open(out / "train.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        (out / "runtime_sec.txt").write_text(f"{time.time() - t0:.2f}")
        return job, rc, time.time() - t0, False

    print(f"{len(jobs)}개 실행, 프로세스 {args.workers}개", flush=True)
    failed = []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            (meta_path, mode), rc, sec, skipped = fut.result()
            if rc:
                failed.append((meta_path.name, mode, rc))
            state = "건너뜀" if skipped else ("실패 rc=" + str(rc) if rc else "OK")
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {mode:8} {meta_path.name.removesuffix('.meta.json')} "
                  f"{state} ({sec / 60:.1f}분)", flush=True)
    if failed:
        print(f"실패 {len(failed)}개: {failed}", flush=True)
        return 1
    return 0


_W: dict = {}


def _control_init(dataset: str) -> None:
    import torch  # noqa: F401  (lightgbm보다 먼저. runner가 lightgbm을 싣지는 않지만 순서를 고정해 둔다)

    from labelbudget.evaluation.runner import FeatureStore

    df = pd.read_csv(ROOT / "data" / "processed" / f"{dataset}.csv")
    _W.update(df=df, graphs=FeatureStore.build(df, cache=ROOT / "data" / "processed" / f"{dataset}_descriptors.csv").graphs)


def _control_one(meta_path: Path) -> tuple[str, float]:
    import numpy as np

    from labelbudget.models.gnn import DMPNNEstimator

    meta, stem = json.loads(meta_path.read_text()), meta_path.name.removesuffix(".meta.json")
    out = meta_path.parents[1] / "runs" / "control" / stem
    if (out / "model_0" / "test_predictions.csv").exists():
        return stem, 0.0
    sp = json.loads(meta_path.with_name(f"{stem}.json").read_text())[0]
    df, graphs, y = _W["df"], _W["graphs"], _W["df"].label.to_numpy()
    tr, va, te = (np.array(sp[k]) for k in ("train", "val", "test"))
    t0 = time.time()
    est = DMPNNEstimator("regression", meta["c_params"], meta["subsample_seed"] + CONTROL_SEED_OFFSET,
                         valid=([graphs[i] for i in va], y[va]), **meta["gnn"]).fit([graphs[i] for i in tr], y[tr])
    pred = est.predict([graphs[i] for i in te])
    (out / "model_0").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"smiles": df.smiles.to_numpy()[te], "label": pred}).to_csv(
        out / "model_0" / "test_predictions.csv", index=False)
    (out / "info.json").write_text(json.dumps({"epochs_run": len(est.history), "best_epoch": est.best_epoch,
                                               "seed": meta["subsample_seed"] + CONTROL_SEED_OFFSET}))
    (out / "runtime_sec.txt").write_text(f"{time.time() - t0:.2f}")
    return stem, time.time() - t0


def control(args) -> int:
    from concurrent.futures import ProcessPoolExecutor

    metas = sorted(_work(args.dataset).glob("splits/*.meta.json"),
                   key=lambda m: -_budget_key(json.loads(m.read_text())["n_labels"]))
    print(f"{len(metas)}개 재학습, 프로세스 {args.workers}개 (초기화 시드 +{CONTROL_SEED_OFFSET})", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_control_init, initargs=(args.dataset,)) as ex:
        for i, fut in enumerate(as_completed([ex.submit(_control_one, m) for m in metas]), 1):
            stem, sec = fut.result()
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(metas)} {stem} ({sec / 60:.1f}분)", flush=True)
    return 0


def collect(args) -> int:
    import numpy as np

    from labelbudget.evaluation.metrics import regression_metrics

    work = _work(args.dataset)
    df = pd.read_csv(ROOT / "data" / "processed" / f"{args.dataset}.csv")
    rows, pred_dir = [], ROOT / "reports" / "results" / "preds" / f"{args.dataset}_chemprop"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for meta_path in sorted(work.glob("splits/*.meta.json")):
        meta, stem = json.loads(meta_path.read_text()), meta_path.name.removesuffix(".meta.json")
        te = np.array(json.loads(meta_path.with_name(f"{stem}.json").read_text())[0]["test"])
        for mode in args.modes:
            out = work / "runs" / mode / stem
            pr = out / "model_0" / "test_predictions.csv"
            if not pr.exists():
                if not (mode in ABLATIONS and meta["n_labels"] not in ("3000", "full")):
                    print(f"없음: {mode} {stem}")
                continue
            p = pd.read_csv(pr)
            if p.smiles.tolist() != df.smiles.iloc[te].tolist():  # 예측 순서가 test 인덱스 순서와 같아야 한다
                raise SystemExit(f"test 순서 불일치: {mode} {stem}")
            pred = p.label.to_numpy(dtype=float)
            if mode == "control":
                epochs_run = json.loads((out / "info.json").read_text())["epochs_run"]
            else:
                epochs_run = int(pd.read_csv(out / "model_0" / "trainer_logs" / "version_0" / "metrics.csv").epoch.max()) + 1
            base = {"dataset": args.dataset, "task": "regression", "target_task": "label",
                    "model": MODEL_NAME[mode], "split_strategy": meta["split_strategy"],
                    "split_seed": meta["split_seed"], "n_labels": meta["n_labels"],
                    "subsample_seed": meta["subsample_seed"], "valid_policy": "full",
                    "n_train": meta["n_train"], "n_valid": meta["n_valid"], "n_test": meta["n_test"],
                    "hp_n_trials": args.n_trials if mode in ("matched", "control") else 0,
                    "hp_selected": json.dumps(meta["c_params"] if mode in ("matched", "control")
                                              else f"chemprop_{mode}"),
                    "epochs_run": epochs_run,
                    "runtime_sec": float((out / "runtime_sec.txt").read_text())}
            rows += [{**base, "metric_name": m, "metric_value": v}
                     for m, v in regression_metrics(df.label.to_numpy()[te], pred).items()]
            pd.DataFrame({"inchikey": df.inchikey.to_numpy()[te], "y_true": df.label.to_numpy()[te],
                          "pred": pred}).to_csv(pred_dir / f"{MODEL_NAME[mode]}_{stem}.csv", index=False)
    out_csv = ROOT / "reports" / "results" / (args.out or f"{args.dataset}_chemprop_full.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"→ {out_csv.relative_to(ROOT)} ({len(rows)}행)\n→ {pred_dir.relative_to(ROOT)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("export", "run", "control", "collect"):
        s = sub.add_parser(name)
        s.add_argument("--dataset", required=True)
        s.add_argument("--n-trials", type=int, default=20, help="matched 모드가 가져올 C의 탐색 체크포인트")
        if name == "export":
            s.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
            s.add_argument("--strategies", nargs="+", default=["scaffold_random"])
        else:
            s.add_argument("--modes", nargs="+", choices=MODES, default=["default", "matched", "control"])
        if name in ("run", "control"):
            s.add_argument("--workers", type=int, default=6)
        if name == "run":
            s.add_argument("--budgets", nargs="+", help="이 예산만 실행 (예: 3000 full). 생략하면 전부")
        if name == "collect":
            s.add_argument("--out", help="reports/results/ 아래 파일 이름 (기본 <dataset>_chemprop_full.csv)")
    args = p.parse_args()
    return {"export": export, "run": run, "control": control, "collect": collect}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
