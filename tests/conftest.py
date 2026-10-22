# torch와 LightGBM은 각자 libomp를 싣는다. LightGBM이 먼저 로드된 프로세스에서는 torch 학습이,
# torch가 먼저면 LightGBM n_jobs>1 학습이 세그폴트로 죽는다. 테스트는 LightGBM을 n_jobs=1로만 쓰므로
# torch를 먼저 로드해 GNN 테스트를 살린다. 실행 스크립트는 베이스라인과 GNN을 다른 프로세스로 돌린다.
try:
    import torch  # noqa: F401
except ImportError:
    pass
