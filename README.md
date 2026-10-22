# LabelBudget

> 분자 물성 예측에서 자기지도 사전학습의 이득은 라벨 몇 개까지 유효한가?

**상태: 설계 완료 · 1주차(데이터 큐레이션) 구현 중. 아직 실험 결과는 없다.**

## 이 프로젝트가 보여주려는 것

1. **딥러닝을 처음부터 구현한다** — D-MPNN을 `scatter` 연산으로 직접 구현하고 PyG 구현과 수치 등가성(차이 < 1e-5)을 테스트로 검증한다. 자기지도 사전학습 2종(masked atom, contrastive)을 직접 구현한다.
2. **그것이 언제 가치 있는지 공정하게 판단한다** — 모든 모델에 동일한 하이퍼파라미터 탐색 예산, scaffold 분할, 시드 10개 이상, Wilcoxon signed-rank 검정.
3. **데이터를 의심한다** — 용해도 과제에서 염 제거가 라벨을 파괴함을 실측으로 확인하고 표준화 정책을 바꿨다. 측정 노이즈 하한을 결과 해석에 반영한다.

사전학습이 이기지 않는 결과도 그대로 보고한다.

## 실험 설계

| ID | 모델 |
|---|---|
| A | ECFP4 + LightGBM |
| B | RDKit 2D 서술자 + LightGBM |
| C | GNN 처음부터 학습 |
| D | GNN + 자기지도 사전학습 (C와 동일 백본) |

- 라벨 예산: 회귀 `100 → 300 → 1K → 3K → 전체`, 분류 `300 → 1K → 3K → 전체` (테스트셋 고정)
- 분할: `random`, `scaffold`, `scaffold_random`
- 데이터: AqSolDB(회귀 주), ESOL(회귀 대조), Tox21 SR-ARE(분류)

핵심 산출물은 B와 D의 성능 곡선이 교차하는 라벨 수와, 그 교차점이 분할 전략에 따라 이동하는지다.

## 설치

```bash
pip install -e ".[dev]"
```

AqSolDB는 자동 다운로드가 불가하다. `curated-solubility-dataset.csv`를 `data/raw/aqsoldb.csv`로 배치한다 (출처: `docs/design/02_IMPLEMENTATION_SPEC.md` §1.2).

## 구조

```
configs/      실험 설정 (결과에 설정 해시를 남긴다)
data/         raw · processed (커밋하지 않음)
docs/design/  연구 기획서 · 구현 스펙 · AqSolDB 감사
src/labelbudget/
  curate/     표준화 · 중복/충돌 · 데이터 카드
  data/       레지스트리 · 로더
  featurize/  ECFP4 · RDKit 서술자
  splits/     분할 전략
  evaluation/ 지표 · 통계 · 라벨 예산 · 실행기
  models/     베이스라인 GBM
scripts/      CLI 진입점
tests/
reports/      그림 · 결과표
```

## 설계 문서

| 문서 | 내용 |
|---|---|
| [`00_HANDOVER.md`](docs/design/00_HANDOVER.md) | 진입점, 핵심 주의사항 |
| [`01_RESEARCH_PLAN.md`](docs/design/01_RESEARCH_PLAN.md) | 연구 질문, 가설, 12주 일정 |
| [`02_IMPLEMENTATION_SPEC.md`](docs/design/02_IMPLEMENTATION_SPEC.md) | 1~3주차 구현 스펙, 수락 기준 |
| [`03_AQSOLDB_AUDIT.md`](docs/design/03_AQSOLDB_AUDIT.md) | 염 처리 정책을 바꾼 실측 감사 |
