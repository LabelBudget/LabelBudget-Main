#!/bin/zsh
# D 단계 자동 연결 (05 §5). 사전학습(MPS)과 그리드(CPU)를 겹쳐 돌린다.
#   A: 마스킹 → 대조학습 사전학습. 도는 동안 그리드 동시 실행 수 4, 끝나면 6
#   B: 실행 중인 C_fixed 대기열이 끝나면 D_mask → D_contrast → C_dmpnn_v2(random) 그리드. 각 가중치가 생길 때까지 기다린다
# 사용법: PYTHON=/경로/python scripts/pipeline_d.sh
cd "$(dirname $0)/.."
PY=${PYTHON:-python}
mkdir -p logs
(
  echo 4 > logs/workers
  for m in mask contrast; do
    $PY scripts/pretrain.py --method $m > logs/pretrain_$m.log 2>&1 || { echo "사전학습 $m 실패" >> logs/pipeline.log; exit 1; }
    echo "[$(date '+%m-%d %H:%M')] 사전학습 $m 완료" >> logs/pipeline.log
  done
  echo 6 > logs/workers
) &
while pgrep -f "queue_d.sh C_fixed" > /dev/null; do sleep 60; done
echo "[$(date '+%m-%d %H:%M')] C_fixed 대기열 종료" >> logs/pipeline.log
for M in D_mask D_contrast; do
  W=models/pretrained/$M.pt
  until [ -f $W ]; do sleep 60; done
  PYTHON=$PY scripts/queue_d.sh $M > logs/$M.log 2>&1
  echo "[$(date '+%m-%d %H:%M')] $M 대기열 종료 (rc=$?)" >> logs/pipeline.log
done
PYTHON=$PY scripts/queue_d.sh C_dmpnn_v2 random > logs/C_dmpnn_v2_random.log 2>&1
echo "[$(date '+%m-%d %H:%M')] C_dmpnn_v2 random 대기열 종료 (rc=$?)" >> logs/pipeline.log
wait
