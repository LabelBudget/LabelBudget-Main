#!/bin/zsh
# D 단계 그리드 대기열 (05 §5). 사용법: scripts/queue_d.sh MODEL [STRATEGY]
# 예산 3000·full은 탐색 10회, 그 이하는 20회 (05 §2 T1). 동시 실행 수는 logs/workers 파일로 실행 중에 바꾼다
set -e
cd "$(dirname $0)/.."
PY=${PYTHON:-python}  # 다른 환경이면 PYTHON=/경로/python scripts/queue_d.sh ...
MODEL=$1; STRAT=${2:-scaffold_random}
case $MODEL in C_fixed) TAG=fixed;; D_mask) TAG=dmask;; D_contrast) TAG=dcontrast;; C_dmpnn_v2) TAG=v2;; *) TAG=$MODEL;; esac
SUFFIX=$([ $STRAT = scaffold_random ] && echo "" || echo "_$STRAT")
for DS in aqsoldb tox21_SR-ARE; do
  OUT=${DS}_gnn_${TAG}${SUFFIX}_full
  $PY scripts/run_parallel.py --dataset $DS --models $MODEL --strategies $STRAT --workers 6 \
      --workers-file logs/workers --n-trials 20 --trials-for 3000=10 full=10 \
      --out reports/results/$OUT.csv --save-preds reports/results/preds/$OUT
done
