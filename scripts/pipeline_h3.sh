#!/bin/zsh
# D 이후 계산 자동 연결 (05 §2.2 U1~U3). pipeline_d.sh가 끝난 뒤 실행된다.
#   1. analyze_d (scaffold_random) → 데이터셋마다 valid 규칙으로 "더 나은 D" (reports/results/<ds>_best_d.json)
#   2. 데이터셋마다 그 D로 random 분할 그리드 (H3)
#   3. AqSolDB 불확실성 재학습 (C v2 + 더 나은 D)
#   4. 앱용 D 모델 학습
# 사용법: PYTHON=/경로/python scripts/pipeline_h3.sh
cd "$(dirname $0)/.."
PY=${PYTHON:-python}
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> logs/pipeline.log; }
while pgrep -f "pipeline_d.sh" > /dev/null; do sleep 60; done
log "pipeline_h3 시작"
for DS in aqsoldb tox21_SR-ARE; do
  $PY scripts/analyze_d.py --dataset $DS > logs/analyze_d_$DS.log 2>&1 || { log "analyze_d $DS 실패"; exit 1; }
  BEST=$($PY -c "import json; print(json.load(open('reports/results/${DS}_best_d.json'))['best_d'])")
  log "$DS 더 나은 D (valid 규칙) = $BEST"
  case $BEST in D_mask) TAG=dmask;; D_contrast) TAG=dcontrast;; esac
  OUT=${DS}_gnn_${TAG}_random_full
  $PY scripts/run_parallel.py --dataset $DS --models $BEST --strategies random --workers 6 \
      --workers-file logs/workers --n-trials 20 --trials-for 3000=10 full=10 \
      --out reports/results/$OUT.csv --save-preds reports/results/preds/$OUT > logs/${BEST}_random_$DS.log 2>&1
  log "$DS $BEST random 그리드 종료 (rc=$?)"
done
BEST=$($PY -c "import json; print(json.load(open('reports/results/aqsoldb_best_d.json'))['best_d'])")
$PY scripts/run_uncertainty.py --models C_dmpnn_v2 $BEST --workers 6 > logs/uncertainty.log 2>&1
log "불확실성 재학습 종료 (rc=$?)"
$PY scripts/train_app_models.py --model $BEST > logs/app_model.log 2>&1
log "앱용 $BEST 모델 학습 종료 (rc=$?)"
