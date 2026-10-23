"""run_baselines.py를 (분할, 시드, 예산) 조합 단위로 쪼개 프로세스 여러 개로 돌리고 결과를 합친다. 결정 G5.

python scripts/run_parallel.py --dataset aqsoldb --models C_dmpnn --strategies scaffold_random --workers 6

- 조합마다 별도 CSV(parts/)에 쓰므로 동시 쓰기 충돌이 없다. 중단 후 다시 실행하면 각 조합이 이어서 돈다
- 오래 걸리는 큰 예산부터 배정해 마지막에 한 프로세스만 남는 시간을 줄인다
- GNN의 스레드 수는 configs gnn.threads가 정한다 (bench: 6×1스레드가 1×8스레드보다 처리량 ~3.1배)
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labelbudget.evaluation.runner import KEY_COLS, MODELS  # noqa: E402  (torch·lightgbm을 싣지 않는다)


def _budget_key(b: str) -> int:
    return 10**9 if b == "full" else int(b)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument("--models", nargs="+", choices=list(MODELS), required=True)
    p.add_argument("--strategies", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--budgets", nargs="+", help="정수 또는 full")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    conf = yaml.safe_load(args.config.read_text())
    df = pd.read_csv(ROOT / "data" / "processed" / f"{args.dataset}.csv", usecols=["label"])
    task = "classification" if set(df.label.unique()) <= {0, 1} else "regression"
    strategies = args.strategies or conf["splits"]["strategies"]
    seeds = args.seeds if args.seeds is not None else conf["seeds"]["split_seeds"]
    budgets = args.budgets or ["full" if b is None else str(b) for b in conf["budgets"][task]]
    policy = conf["hp_search"].get("valid_policy", "full")
    group = "gnn" if any(MODELS[m].kind == "gnn" for m in args.models) else "baselines"
    out = args.out or ROOT / "reports" / "results" / f"{args.dataset}_{group}_{policy}.csv"
    parts = out.parent / "parts" / out.stem
    parts.mkdir(parents=True, exist_ok=True)

    jobs = sorted(((s, k, b) for s in strategies for k in seeds for b in budgets),
                  key=lambda j: (-_budget_key(j[2]), j[0], j[1]))

    def run(job):
        s, k, b = job
        stem = parts / f"{s}_s{k}_n{b}"
        cmd = [sys.executable, str(ROOT / "scripts" / "run_baselines.py"), "--dataset", args.dataset,
               "--config", str(args.config), "--models", *args.models, "--strategies", s, "--seeds", str(k),
               "--budgets", b, "--n-jobs", "1", "--out", str(stem.with_suffix(".csv"))]
        t0 = time.time()
        with open(stem.with_suffix(".log"), "a") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        return job, rc, time.time() - t0

    print(f"{len(jobs)}개 조합, 프로세스 {args.workers}개 → {parts}", flush=True)
    failed = []
    with ThreadPoolExecutor(args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(run, j) for j in jobs]), 1):
            (s, k, b), rc, sec = fut.result()
            if rc:
                failed.append((s, k, b, rc))
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} {s} s{k} n={b} "
                  f"{'실패 rc=' + str(rc) if rc else 'OK'} ({sec / 60:.1f}분)", flush=True)

    frames = [pd.read_csv(f, dtype=str) for f in sorted(parts.glob("*.csv"))]
    if out.exists():
        frames.insert(0, pd.read_csv(out, dtype=str))
    merged = pd.concat(frames).drop_duplicates(KEY_COLS + ["hp_n_trials", "metric_name"], keep="first")
    merged.to_csv(out, index=False)
    print(f"→ {out} ({len(merged)}행)", flush=True)
    if failed:
        print(f"실패 {len(failed)}개: {failed} — 같은 명령으로 다시 실행하면 이어서 돈다", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
