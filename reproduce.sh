#!/usr/bin/env bash
# 전체 재현 진입점. 각 단계는 구현되는 대로 채운다.
set -euo pipefail

# 1. 데이터 다운로드 (AqSolDB는 data/raw/aqsoldb.csv 수동 배치)
# python scripts/download_data.py

# 2. 감사 — docs/design/03 수치 재현
# python scripts/audit_dataset.py --dataset aqsoldb --out docs/audit_aqsoldb.md
# python scripts/audit_dataset.py --dataset tox21 --task SR-ARE --out docs/audit_tox21.md

# 3. 큐레이션 → 데이터 카드
# python scripts/curate_dataset.py --dataset aqsoldb
# python scripts/curate_dataset.py --dataset esol
# python scripts/curate_dataset.py --dataset tox21 --task SR-ARE

# 4. 베이스라인 A·B
# python scripts/run_baselines.py --config configs/default.yaml

echo "아직 구현된 단계가 없습니다." >&2
exit 1
