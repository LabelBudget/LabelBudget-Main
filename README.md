# LabelBudget

> 분자 물성 예측에서 자기지도 사전학습의 이득은 라벨 몇 개까지 유효한가?

**🔗 웹 앱: [labelbudget.streamlit.app](https://labelbudget.streamlit.app)** — 라벨 예산 곡선·시드별 비교·심화 분석을 직접 탐색하고, SMILES로 용해도를 예측해 볼 수 있다.

**결론: 이 설정에서는 라벨이 몇 개이든 유효하지 않았다.**
서술자 + LightGBM(B)이 라벨 100개부터 전체까지 가장 강하거나 같았다. ZINC250k로 사전학습한 D-MPNN(D)은 용해도(AqSolDB)에서 같은 크기의 사전학습 없는 모델보다 오히려 나빴고(부정적 전이), 독성(Tox21 SR-ARE)에서는 효과가 없었다.
판정 규칙은 결과를 보기 전에 커밋했고([`05_PRETRAIN_SPEC.md`](docs/design/05_PRETRAIN_SPEC.md) §4, 커밋 `3f50a0f`), 부정적 결과를 그대로 보고한다.

![AqSolDB 라벨 예산 곡선](reports/figures/aqsoldb_budget_curve_d.png)

## 이 프로젝트가 보여주려는 것

1. **딥러닝을 처음부터 구현한다** — D-MPNN을 torch 기본 연산(`scatter`)으로 직접 구현하고, 공식 chemprop 2.3.1과 출력이 비트 단위로 같음(15,326분자, max|Δ| = 0)을 테스트로 검증했다. 자기지도 사전학습 2종(원자 마스킹, GraphCL 대조학습)도 직접 구현했다.
2. **그것이 언제 가치 있는지 공정하게 판단한다** — 모든 모델에 같은 하이퍼파라미터 탐색 예산, scaffold 분할, 시드 10개, 시드 쌍 Wilcoxon + Holm 보정. 판정 규칙은 결과 전에 고정했다.
3. **데이터를 의심한다** — 용해도 과제에서 염 제거가 라벨을 파괴함을 실측으로 확인하고 표준화 정책을 바꿨다([`03_AQSOLDB_AUDIT.md`](docs/design/03_AQSOLDB_AUDIT.md)). 측정 노이즈 하한(출처 간 SD 0.30 logS)과 라벨 신뢰도를 결과 해석에 반영했다.

## 핵심 결과

| | AqSolDB (용해도 회귀, RMSE ↓) | Tox21 SR-ARE (독성 분류, AUPRC ↑) |
|---|---|---|
| 서술자(B) vs 지문(A) | B가 모든 분할·예산에서 우세 (10/10 시드) | 같음 |
| GNN 처음부터(C v2) vs B | 유의 차 없음. 1,000개 이하는 B, 3,000개 이상은 C 쪽 | 모든 예산에서 B ≥ C, 3,000개 유의 |
| **사전학습 D − 같은 크기 C_fixed** | **D가 나쁨**: 마스킹 +0.06~+0.16 (300개~전체 유의), 대조학습 +0.03~+0.11 (300~3,000개 유의) | 차이 없음 (모든 예산 Holm p = 1.0) |
| D vs B (교차점, H2) | D가 이기는 예산 없음. 100개부터 B 우세 | D가 이기는 예산 없음 |
| 분할을 바꾸면 교차점이 이동하는가 (H3) | 교차점이 없어 이동도 없음 | 같음 |

모든 비교는 scaffold_random 분할(주 분석), 시드 10개 쌍 Wilcoxon, 예산 간 Holm 보정이다.

**추가 분석**

- **왜 나빴나 (사후 진단, 사전 등록 아님)** — 인코더를 고정한 선형 탐침에서도 사전학습 표현이 무작위 초기화보다 나빴다(전체 예산 RMSE 1.420 vs 대조학습 1.450 · 마스킹 1.603). 버그가 아니라 부정적 전이로 해석한다 → [`d_diagnosis_aqsoldb.md`](reports/d_diagnosis_aqsoldb.md)
- **라벨 신뢰도** — 출처가 하나뿐인 분자는 복수 출처 분자보다 RMSE가 0.13~0.27 크다. 같은 층 안의 모델 간 차이(≤ 0.04)보다 크다 → [`reliability_aqsoldb.md`](reports/reliability_aqsoldb.md)
- **불확실성** — deep ensemble·MC dropout의 90% 구간은 실제로 21~43%만 포함한다(크게 과신, 오차와의 순위 상관 ≤ 0.11). split conformal은 83~88%로 거의 맞지만 폭이 3~4.5 logS로 넓다 → [`uncertainty_aqsoldb.md`](reports/uncertainty_aqsoldb.md)

결론의 범위와 한계는 [`LIMITATIONS.md`](LIMITATIONS.md)에 있다. 특히 사전학습은 방법당 1회·문헌 기본 HP·25만 개 규모이고, 미세조정 방식은 하나뿐이다.

## 모델

| ID | 모델 | 비고 |
|---|---|---|
| A | ECFP4(2048) + LightGBM | 바닥 기준 |
| B | RDKit 2D 서술자 + LightGBM | 가장 강한 저비용 기준선 |
| C v2 | D-MPNN 처음부터 학습 (크기·readout 탐색) | 직접 구현 |
| C_fixed | C와 같되 크기 고정 (d_h 300, depth 3) | 사전학습 효과만 분리하는 대조군 |
| D_mask / D_contrast | C_fixed 백본 + ZINC250k 자기지도 사전학습 → 전체 미세조정 | 원자 마스킹 / GraphCL 대조학습 |

- 라벨 예산: 회귀 `100 → 300 → 1K → 3K → 전체`, 분류 `300 → 1K → 3K → 전체` (test 고정)
- HP 탐색: 라벨 ≤1,000개 20회, ≥3,000개 10회. 모든 모델을 같은 체크포인트로 비교
- 학습 조합 약 2,050개, 실패 0. 미세조정은 CPU 결정론 실행

## 검증과 재현성

- **공식 구현과 대조** — 같은 분할에서 공식 chemprop을 돌려 직접 구현을 검증했다. 그 과정에서 C의 readout 문제를 찾아 v2로 고쳤다(test를 본 뒤의 변경이라 v1도 함께 보고)
- **사전 등록** — D 설계, 비교, 가설 판정, "더 나은 D" 선택 규칙(valid만 사용)을 결과 전에 커밋했다
- **누수 차단** — 평가셋과 골격이 같은 ZINC 분자 115개를 사전학습 데이터에서 뺐다. 마스킹이 가린 원자 정보를 흘리지 않는지 테스트한다
- **재현 확인** — 불확실성 실험의 0번 모델과 앱 모델이 주 그리드의 test RMSE를 그대로 재현하는지 스크립트가 확인한다
- **테스트** — `pytest` 102개 통과 (chemprop 등가성은 별도 환경)

## 웹 앱: 결과 탐색 + 용해도 예측

| 페이지 | 내용 |
|---|---|
| 개요 | 질문, 결론, 사전 등록 가설 판정 |
| 라벨 예산 곡선 | 데이터셋·분할·모델을 골라 곡선 보기, 사전 등록 비교의 시드별 차이와 Holm p |
| 심화 분석 | 사후 진단(선형 탐침), 라벨 신뢰도 층화, 불확실성 구간의 실제 포함률 |
| 용해도 예측 | SMILES → B와 D 예측, D의 conformal 90% 구간, 가장 비슷한 학습 분자 3개와 실측값, 적용 범위 경고 |

![라벨 예산 곡선 페이지](reports/figures/app_curves.png)
![용해도 예측 페이지](reports/figures/app_demo.png)

```bash
conda activate labelbudget-app      # streamlit 전용 환경 (본 환경과 분리)
streamlit run app/streamlit_app.py  # http://localhost:8501
```

- 결과 페이지는 `app/data/*.csv`만 읽는다 (`python scripts/build_app_data.py`로 결과 파일에서 다시 만든다)
- 배포 전 점검: `python scripts/check_app.py` (앱 환경). 모든 페이지와 예시 분자 26개를 실행한다
- Streamlit Community Cloud 배포: main file `app/streamlit_app.py`, Python 3.13. 의존성은 `app/requirements.txt`(CPU torch)·`packages.txt`(LightGBM·RDKit 그림용 시스템 라이브러리)
- 예측은 시연용이다. 사전학습 모델(D)이 서술자 모델(B)보다 낫지 않다 ([`model_card.md`](docs/model_card.md))

## 설치

```bash
conda env create -f environment.yml
conda activate labelbudget
```

conda 없이 설치하려면 `pip install -e ".[dev]"`. macOS에서는 LightGBM이 `libomp`를 요구하므로 `brew install libomp`가 필요하다.
AqSolDB는 자동 다운로드가 불가하다. `curated-solubility-dataset.csv`를 `data/raw/aqsoldb.csv`로 배치한다 (출처: `docs/design/02_IMPLEMENTATION_SPEC.md` §1.2).

## 재현

```bash
python scripts/curate_dataset.py --dataset aqsoldb
python scripts/run_baselines.py --dataset aqsoldb --n-jobs 8
python scripts/run_parallel.py --dataset aqsoldb --models C_dmpnn_v2 --strategies scaffold_random --workers 6 --out reports/results/aqsoldb_gnn_v2_full.csv
python scripts/curate_pretrain.py
python scripts/pretrain.py --method mask && python scripts/pretrain.py --method contrast   # 또는 커밋된 가중치 사용
scripts/queue_d.sh C_fixed && scripts/queue_d.sh D_mask && scripts/queue_d.sh D_contrast
python scripts/analyze_d.py --dataset aqsoldb
scripts/pipeline_h3.sh                     # random 분할 → 불확실성 → 앱 모델
python scripts/analyze_d.py --dataset aqsoldb --split random
python scripts/analyze_reliability.py --best-d D_contrast
python scripts/analyze_uncertainty.py
pytest
```

미세조정은 CPU 결정론이라 같은 장치·같은 스레드 수에서 결과가 그대로 재현된다. 사전학습(MPS)은 비결정론이므로 가중치를 SHA-256과 함께 커밋했다 (`models/pretrained/`).

## 구조

```
configs/      실험 설정 (결과에 설정 해시를 남긴다)
data/         raw · processed (커밋하지 않음)
docs/         데이터 카드 · 모델 카드 · 설계 문서
src/labelbudget/
  curate/     표준화 · 중복/충돌 · 데이터 카드
  featurize/  ECFP4 · RDKit 서술자 · 분자 그래프
  splits/     분할 전략
  evaluation/ 지표 · 통계 · 라벨 예산 · 실행기
  models/     GBM · D-MPNN
  pretrain/   마스킹 · 대조학습 과제
scripts/      CLI 진입점
app/          Streamlit 웹 앱 (views/ 페이지, data/ 앱용 결과표)
models/       사전학습 가중치 · 앱 모델
reports/      결과표 · 그림 · 분석 보고서
```

## 보고서

| 보고서 | 내용 |
|---|---|
| [`d_aqsoldb.md`](reports/d_aqsoldb.md), [`d_tox21_SR-ARE.md`](reports/d_tox21_SR-ARE.md) | D 주 분석, 사전 등록 판정 |
| [`d_aqsoldb_random.md`](reports/d_aqsoldb_random.md), [`d_tox21_SR-ARE_random.md`](reports/d_tox21_SR-ARE_random.md) | H3 random 분할 |
| [`d_diagnosis_aqsoldb.md`](reports/d_diagnosis_aqsoldb.md) | 사후 진단 (선형 탐침) |
| [`reliability_aqsoldb.md`](reports/reliability_aqsoldb.md) | 라벨 신뢰도 층화 |
| [`uncertainty_aqsoldb.md`](reports/uncertainty_aqsoldb.md) | 불확실성 비교 |
| [`baselines_aqsoldb.md`](reports/baselines_aqsoldb.md), [`gnn_v2_aqsoldb.md`](reports/gnn_v2_aqsoldb.md), [`chemprop_aqsoldb.md`](reports/chemprop_aqsoldb.md) | 베이스라인, C, chemprop 재현 |

## 설계 문서

| 문서 | 내용 |
|---|---|
| [`00_HANDOVER.md`](docs/design/00_HANDOVER.md) | 진입점, 핵심 주의사항 |
| [`01_RESEARCH_PLAN.md`](docs/design/01_RESEARCH_PLAN.md) | 연구 질문, 가설, 일정 |
| [`02_IMPLEMENTATION_SPEC.md`](docs/design/02_IMPLEMENTATION_SPEC.md) | 구현 스펙, 수락 기준 |
| [`03_AQSOLDB_AUDIT.md`](docs/design/03_AQSOLDB_AUDIT.md) | 염 처리 정책을 바꾼 실측 감사 |
| [`05_PRETRAIN_SPEC.md`](docs/design/05_PRETRAIN_SPEC.md) | D 설계, 결과 전 고정한 판정 규칙, 결과 |
