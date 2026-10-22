"""데이터셋 다운로드. 스펙 02 §1."""
import argparse
import sys

from labelbudget.data.loaders import DatasetUnavailable, download, load_raw
from labelbudget.data.sources import SOURCES


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(SOURCES), action="append", help="생략하면 전체")
    p.add_argument("--force", action="store_true", help="캐시가 있어도 다시 받는다")
    args = p.parse_args()

    failed = 0
    for name in args.dataset or sorted(SOURCES):
        try:
            path = download(name, force=args.force)
            df = load_raw(name, allow_download=False)
            print(f"[ok]   {name:<14} {df.shape}  {path}")
        except DatasetUnavailable as e:
            failed += 1
            print(f"[fail] {name}\n{e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
