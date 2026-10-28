# Model card — LabelBudget 데모 모델 (수용해도 logS)

> 상태: 확정 (2026-10-28). 성능은 `reports/d_aqsoldb.md`(scaffold_random 주 분석), 불확실성은 `reports/uncertainty_aqsoldb.md`에서 옮겼다.

## 1. 무엇인가

- 연구 프로젝트 LabelBudget("자기지도 사전학습의 이득은 라벨 몇 개까지 유효한가")의 **시연용 모델 2개**
  - **B**: RDKit 2D 서술자(217개 중 학습셋 기준 선택) + LightGBM
  - **D**: D-MPNN(d_h 300, depth 3) 인코더를 ZINC250k로 자기지도 사전학습한 뒤 미세조정. 마스킹·대조학습 중 valid 규칙(05 §2.2 U1)으로 고른 1개
- 앱: `app/streamlit_app.py`. 학습: `scripts/train_app_models.py`

## 2. 학습 데이터

- AqSolDB 큐레이션본 8,703개 (`docs/data_card_aqsoldb.md`). 앱 모델은 scaffold_random 분할 시드 0의 학습셋 6,962개로 학습
- HP는 주 그리드가 valid로 고른 값(탐색 10회 체크포인트)을 그대로 쓴다. test 분자는 학습에 쓰지 않았다
- D의 사전학습 데이터: ZINC250k 249,219개 (평가 데이터와 골격이 같은 115개 제거, `docs/data_card_zinc250k_pretrain.md`)

## 3. 의도한 용도

- 연구 결과를 손으로 만져 보는 시연: 같은 분자에 대해 서술자 모델과 사전학습 GNN이 어떻게 다르게 예측하는지 본다

## 4. 쓰면 안 되는 경우

- **실험값을 대신하는 용도**: 의약·환경·안전 판단에 쓰지 않는다
- **학습 분포 밖 분자**: 최근접 학습 분자와 ECFP4 Tanimoto < 0.4이면 앱이 경고한다. 이 기준은 관행적 값이며 검증하지 않았다
- **ZINC에 없는 원소**(금속, 붕소, 규소 등): 사전학습에서 본 적이 없다. AqSolDB에서도 드물다 (붕소 케이지 분자 사례, `LIMITATIONS.md`)
- **염·혼합물**: 표준화 단계에서 다성분 분자는 거부한다 (용해도는 염을 떼면 값이 바뀐다)
- **입체이성질체 구분**: 같은 골격의 입체이성질체 차이를 신뢰하지 않는다

## 5. 성능 (AqSolDB scaffold_random test RMSE logS, 시드 10개 평균 ± SD)

valid 규칙(05 §2.2 U1)으로 고른 D는 **D_contrast(대조학습)**다. 비교를 위해 사전학습 없는 C도 함께 적는다.

| 라벨 수 | B 서술자+LGBM | C_fixed (사전학습 없음) | D_contrast (앱 모델) |
|---:|---:|---:|---:|
| 100 | 1.464 ± 0.302 | 1.560 ± 0.242 | 1.636 ± 0.239 |
| 300 | 1.225 ± 0.052 | 1.285 ± 0.078 | 1.356 ± 0.080 |
| 1,000 | 1.153 ± 0.037 | 1.157 ± 0.051 | 1.262 ± 0.080 |
| 3,000 | 1.075 ± 0.035 | 1.049 ± 0.064 | 1.121 ± 0.058 |
| 전체 (6,962) | 1.026 ± 0.045 | 0.997 ± 0.037 | 1.023 ± 0.043 |

- **사전학습은 이 데이터에서 도움이 되지 않았다.** D_contrast는 300·1,000·3,000개에서 같은 크기의 사전학습 없는 C_fixed보다 유의하게 나쁘다 (시드 쌍 Wilcoxon, Holm). 앱이 D를 보여 주는 이유는 "사전학습 모델이 실제로 어떻게 예측하는가"를 시연하기 위해서이지, 더 좋은 모델이라서가 아니다
- 사후 진단(사전 등록 아님): 사전학습 표현의 선형 탐침이 무작위 초기화 표현보다 나빴다 (`reports/d_diagnosis_aqsoldb.md`). 부정적 전이로 해석한다
- 측정 노이즈 하한: 복수 출처 분자의 출처 간 표준편차 평균 0.301 logS (`LIMITATIONS.md`)
- 앱 모델 자체(시드 0, 전체 예산, 학습 6,962개)의 test RMSE: B 1.076, D_contrast 1.031 (`models/app/meta.json`, 주 그리드 값과 같음을 학습 스크립트가 확인)
- 단일 출처(AqSolDB Group G1) 분자는 복수 출처보다 RMSE가 0.13~0.27 크다 (`reports/reliability_aqsoldb.md`)

## 6. 불확실성

- **앱은 점 예측만 보여 준다.** 아래 결과 때문에 단일 모델의 신뢰도 표시를 넣지 않았다
- 같은 설정(scaffold_random, 시드 10개)에서 90% 구간의 실제 포함률 (`reports/uncertainty_aqsoldb.md`):
  - deep ensemble 5개: 0.25~0.36, MC dropout: 0.21~0.43 → 크게 과신한다. 표준편차와 |오차|의 Spearman ρ ≤ 0.11
  - split conformal: 0.83~0.88 (목표 0.90), 폭 3.0~4.5 logS → 거의 맞지만 분자 구분 없이 넓다
- 앱 예측을 쓸 때는 test RMSE(약 1.0 logS)와 측정 노이즈 하한(0.3 logS)을 오차 크기의 기준으로 삼는다

## 7. 재현

```bash
python scripts/train_app_models.py --model B_desc_lgbm
python scripts/train_app_models.py --model <reports/results/aqsoldb_best_d.json의 best_d>
streamlit run app/streamlit_app.py
```

학습 스크립트는 test RMSE가 주 그리드 값과 같은지 확인하고, 다르면 실패한다.
