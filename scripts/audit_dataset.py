"""데이터셋 감사 — 03 문서 수치 재현. 스펙 02 §13.

python scripts/audit_dataset.py --dataset aqsoldb --out docs/audit_aqsoldb.md
python scripts/audit_dataset.py --dataset tox21 --task SR-ARE --out docs/audit_tox21.md
"""
import argparse
from pathlib import Path

from labelbudget.curate.audit import POLICIES, audit, render_audit
from labelbudget.data.sources import SOURCES


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    p.add_argument("--task", help="라벨 컬럼 (Tox21 과제 등)")
    p.add_argument("--policy", action="append", choices=list(POLICIES), help="생략하면 4개 전부")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    report = audit(args.dataset, args.task, policies=args.policy, top_n=args.top)
    md = render_audit(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"→ {args.out}")
    for name, r in report["policies"].items():
        print(f"{name:18} 통과 {r.n_pass:,}  고유 {r.n_unique:,}  만든 중복 {r.n_created_groups}")
    if not args.out:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
