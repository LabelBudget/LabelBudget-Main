"""run_baselines.py를 (분할, 시드, 예산) 조합 단위로 쪼개 프로세스 여러 개로 돌리고 결과를 합친다. 결정 G5.

python scripts/run_parallel.py --dataset aqsoldb --models C_dmpnn --strategies scaffold_random --workers 6

- 조합마다 별도 CSV(parts/)에 쓰므로 동시 쓰기 충돌이 없다. 중단 후 다시 실행하면 각 조합이 이어서 돈다
- 오래 걸리는 큰 예산부터 배정해 마지막에 한 프로세스만 남는 시간을 줄인다
- GNN의 스레드 수는 configs gnn.threads가 정한다 (bench: 6×1스레드가 1×8스레드보다 처리량 ~3.1배)
- --trials-for 3000=10 full=10: 예산별 탐색 횟수 (05 §2 T1). 같은 시드면 앞 k회 후보열이 같아 체크포인트 값과 일치한다
- --workers-file: 파일에 적힌 정수로 동시 실행 수를 실행 중에 바꾼다 (사전학습과 겹치는 동안 줄이기). 결과에는 영향이 없다
"""
import argparse
import threading
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
    p.add_argument("--workers-file", type=Path, help="동시 실행 수를 실행 중에 읽어 오는 파일 (없거나 잘못되면 --workers)")
    p.add_argument("--n-trials", type=int, help="기본 탐색 횟수 (없으면 configs)")
    p.add_argument("--trials-for", nargs="+", default=[], metavar="BUDGET=N", help="예산별 탐색 횟수")
    p.add_argument("--out", type=Path)
    p.add_argument("--save-preds", type=Path, help="run_baselines.py --save-preds로 넘긴다")
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

    trials_for = dict(t.split("=") for t in args.trials_for)
    active, cond = [0], threading.Condition()

    def limit() -> int:
        try:
            return max(1, int(args.workers_file.read_text().strip()))
        except (AttributeError, OSError, ValueError):
            return args.workers

    def run(job):
        with cond:
            while active[0] >= limit():
                cond.wait(timeout=30)
            active[0] += 1
        try:
            return _run(job)
        finally:
            with cond:
                active[0] -= 1
                cond.notify_all()

    def _run(job):
        s, k, b = job
        stem = parts / f"{s}_s{k}_n{b}"
        cmd = [sys.executable, str(ROOT / "scripts" / "run_baselines.py"), "--dataset", args.dataset,
               "--config", str(args.config), "--models", *args.models, "--strategies", s, "--seeds", str(k),
               "--budgets", b, "--n-jobs", "1", "--out", str(stem.with_suffix(".csv"))]
        if args.save_preds:
            cmd += ["--save-preds", str(args.save_preds)]
        n = trials_for.get(b, args.n_trials)
        if n:
            cmd += ["--n-trials", str(n)]
        t0 = time.time()
        with open(stem.with_suffix(".log"), "a") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        return job, rc, time.time() - t0

    print(f"{len(jobs)}개 조합, 프로세스 {args.workers}개 → {parts}", flush=True)
    failed = []
    with ThreadPoolExecutor(max(args.workers, 8)) as ex:  # 실제 동시 실행 수는 limit()이 정한다
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
