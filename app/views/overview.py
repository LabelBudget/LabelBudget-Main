"""개요: 질문, 결론, 문제와 데이터 · 최종 성능 · 한계점 · 링크 (README와 같은 내용)."""
import streamlit as st
from common import REPO

st.title("LabelBudget")
st.markdown("#### 분자 물성 예측에서 자기지도 사전학습의 이득은 라벨 몇 개까지 유효한가?")

st.info("**결론: 이 설정에서는 라벨이 몇 개이든 유효하지 않았다.** 서술자 + LightGBM이 라벨 100개부터 전체까지 "
        "가장 강하거나 같았고, 사전학습한 GNN은 용해도에서 같은 크기의 사전학습 없는 GNN보다 오히려 나빴다(부정적 전이). "
        "독성 분류에서는 효과가 없었다. 판정 규칙은 결과를 보기 전에 커밋했다.")

c = st.columns(4)
kw = dict(delta_color="off", delta_arrow="off", border=True)
c[0].metric("학습 조합", "2,050", "실패 0", **kw)
c[1].metric("직접 구현 vs chemprop", "차이 0", "15,326분자 동일", help="같은 입력에서 출력이 비트 단위로 같다", **kw)
c[2].metric("사전학습 효과 · AqSolDB", "나빠짐", "RMSE +0.07~0.16", help="라벨 300개, D − C_fixed, 두 방법 모두 Holm 유의", **kw)
c[3].metric("사전학습 효과 · Tox21", "없음", "Holm p = 1.0", help="모든 예산, D − C_fixed", **kw)

problem, perf, limits, links = st.tabs(["문제와 데이터", "최종 성능", "한계점", "링크"])

with problem:
    st.subheader("해결하려 한 문제")
    st.markdown("""
신약·소재 개발에서 분자 물성 라벨은 실험으로만 얻을 수 있어 비싸다. 그래서 라벨 없는 분자 수십만 개로 GNN을 먼저 학습하는
**자기지도 사전학습**이 대안으로 제안됐지만, 문헌의 답은 엇갈린다. 공정하게 비교하면 이득이 거의 없고 서술자 기반 고전 ML이 여전히
강하다는 보고도 있다.

이 프로젝트는 질문을 "도움이 되는가"에서 **라벨이 몇 개일 때까지 도움이 되는가**로 바꿨다. 라벨 100개부터 전체까지 예산을 늘려 가며
같은 분할·같은 탐색 예산·같은 시드로 비교한다. 실무 질문으로는 "라벨이 N개 있을 때 사전학습 GNN을 만들 가치가 있는가"다.
""")
    st.subheader("데이터 구성")
    st.markdown("""
| 데이터 | 용도 | 과제 | 원본 → 정리 후 | test (분할당) |
|---|---|---|---:|---:|
| **AqSolDB** | 주 평가 | 수용해도 logS 회귀 (−13.2 ~ +2.1) | 9,982 → **8,703** | 871 |
| **Tox21 SR-ARE** | 분류 평가 | 산화 스트레스 반응 독성, 양성 약 16% | 7,831 → **5,514** | 552 (양성 약 112) |
| **ZINC250k** | 사전학습 (라벨 없음) | — | 249,455 → **249,219** | — |

- **정리**: RDKit 표준화, 유기 단일 성분만(염·혼합물 거부 — 용해도는 염을 떼면 라벨이 바뀐다), 중복 병합. 평가셋과 골격이 같은 ZINC 분자 115개는 사전학습에서 제외
- **분할**: scaffold_random(골격 단위 80/10/10, 시드 0~9) — 같은 골격이 학습과 test에 함께 들어가지 않는다. H3용으로 random 분할도 돌렸다
- **라벨 예산**: 회귀 100 · 300 · 1,000 · 3,000 · 전체(6,962), 분류 300 · 1,000 · 3,000 · 전체(4,411). valid·test는 고정
- **라벨 품질**: AqSolDB의 77.6%는 출처가 하나뿐이고, 출처 간 표준편차 평균 0.30 logS가 측정 노이즈 하한이다
""")
    st.subheader("비교한 모델")
    st.markdown("""
| 모델 | 설명 |
|---|---|
| **B** 서술자 + LightGBM | RDKit 2D 서술자 217개. 가장 강한 저비용 기준선 |
| **C v2** D-MPNN | 처음부터 학습. 크기·readout까지 탐색. torch로 직접 구현 |
| **C_fixed** | C와 같되 크기 고정(d_h 300, depth 3). 사전학습 효과만 분리하는 대조군 |
| **D** 마스킹 / 대조학습 | C_fixed 백본을 ZINC250k로 자기지도 사전학습한 뒤 전체 미세조정 |

모든 모델에 같은 탐색 예산(≤1,000개 20회, ≥3,000개 10회), 시드 10개. 비교는 시드 쌍 Wilcoxon + 예산 간 Holm 보정.
""")

with perf:
    st.markdown("scaffold_random 분할, test 기준 시드 10개 평균. **굵게**는 그 행의 최고값. 곡선 전체는 **라벨 예산 곡선** 페이지.")
    st.markdown("""
**AqSolDB (용해도 회귀)**

| 라벨 수 | 지표 | B | C v2 | C_fixed | D 마스킹 | D 대조학습 |
|---:|---|---:|---:|---:|---:|---:|
| 100 | RMSE ↓ | **1.464** | 1.621 | 1.560 | 1.654 | 1.636 |
| 300 | RMSE ↓ | **1.225** | 1.268 | 1.285 | 1.437 | 1.356 |
| 전체 | RMSE ↓ | 1.026 | **0.994** | 0.997 | 1.053 | 1.023 |
| 전체 | R² ↑ | 0.796 | **0.808** | **0.808** | 0.785 | 0.797 |

**Tox21 SR-ARE (독성 분류)** — AUPRC의 무작위 기준선은 test 양성률(약 0.20)

| 라벨 수 | 지표 | B | C v2 | C_fixed | D 마스킹 | D 대조학습 |
|---:|---|---:|---:|---:|---:|---:|
| 300 | AUPRC ↑ | **0.377** | 0.356 | 0.353 | 0.346 | 0.350 |
| 전체 | AUPRC ↑ | **0.538** | 0.506 | 0.477 | 0.473 | 0.491 |
| 전체 | AUROC ↑ | **0.804** | 0.779 | 0.766 | 0.761 | 0.768 |
""")
    st.subheader("사전 등록 가설 판정")
    st.markdown("""
| 가설 | AqSolDB (용해도) | Tox21 SR-ARE (독성) |
|---|---|---|
| **H1** 사전학습 이득은 라벨이 적을수록 크다 | 기각 — 이득이 없고 모든 예산에서 D가 C_fixed보다 나쁜 쪽 | 기각 — 0 근처 |
| **H2** D가 B를 이기다 큰 예산에서 따라잡힌다 (교차점) | 교차점 없음 — 라벨 100개부터 B 우세 | 교차점 없음 — D가 이기는 예산 없음 |
| **H3** 교차점이 scaffold 분할에서 더 오른쪽 | 판정 불가 — 두 분할 모두 교차점 없음 | 판정 불가 — 같음 |
| **H4** 구조 탐색(C v2 − C_fixed) 효과와 함께 보기 | 크기 탐색 효과 없음 → D의 열세는 사전학습 가중치에서 옴 | 같음 |
""")
    st.subheader("추가로 알게 된 것")
    st.markdown("""
- **왜 나빴나** (사후 진단): 인코더를 고정한 선형 탐침에서도 사전학습 표현이 무작위 초기화보다 나빴다 → 버그가 아니라 부정적 전이
- **라벨 신뢰도가 더 큰 지렛대**: 출처가 하나뿐인 분자의 오차가 0.13~0.27 크다. 같은 층 안의 모델 간 차이(≤ 0.04)보다 크다
- **GNN의 자기 확신은 오차를 알려 주지 못한다**: 앙상블·MC dropout의 90% 구간이 실제로 21~43%만 포함한다

자세한 내용은 **심화 분석** 페이지.
""")

with limits:
    st.markdown(f"""
- **사전학습 규모와 횟수** — ZINC250k는 Hu et al.(ZINC 200만 개)의 약 1/8이고, 방법마다 1회만 문헌 기본 HP로 학습했다. 결론은 "이 규모·이 설정에서"로 한정한다
- **원인을 하나로 좁히지 못함** — 미세조정 방식이 하나뿐이라, 부정적 전이가 표현 자체 때문인지 미세조정 방식 때문인지 실험으로 가르지 않았다
- **데이터셋 2개** — 회귀 1개·분류 1개라 다른 물성으로 일반화하기는 이르다
- **라벨 노이즈 하한** — 측정 노이즈(0.30 logS)와 단일 출처 라벨 때문에 test 오차의 일부는 어떤 모델로도 줄일 수 없다
- **test를 본 뒤의 설계 변경 1건** — C의 readout 탐색(v2)은 공식 chemprop과 대조한 test 결과를 보고 추가했다. 사전 등록 결과(v1)도 함께 보고한다
- **범위 축소** — H3는 random 분할만, 큰 예산은 탐색 10회, 불확실성 비교는 AqSolDB만 (기한 때문, 결과 전에 결정)

근거 전체: [LIMITATIONS.md]({REPO}/blob/main/LIMITATIONS.md)
""")

with links:
    st.markdown(f"""
| 종류 | 링크 |
|---|---|
| GitHub 저장소 | [{REPO.removeprefix('https://')}]({REPO}) |
| 데모 | 지금 보고 있는 앱 — 결과 탐색 + SMILES 용해도 예측 |
| 연구 설계 · 결과 전 고정한 판정 규칙 | [05_PRETRAIN_SPEC.md]({REPO}/blob/main/docs/design/05_PRETRAIN_SPEC.md) |
| 결과 보고서 | [d_aqsoldb.md]({REPO}/blob/main/reports/d_aqsoldb.md) · [d_tox21_SR-ARE.md]({REPO}/blob/main/reports/d_tox21_SR-ARE.md) |
| 모델 카드 · 한계 | [model_card.md]({REPO}/blob/main/docs/model_card.md) · [LIMITATIONS.md]({REPO}/blob/main/LIMITATIONS.md) |

**참고 논문**

- Sun et al., *Does GNN Pretraining Help Molecular Representation?* NeurIPS 2022 — [arXiv:2207.06010](https://arxiv.org/abs/2207.06010)
- Deng et al., *A systematic study of key elements underlying molecular property prediction*, Nature Communications 2023 — [링크](https://www.nature.com/articles/s41467-023-41948-6)
- Hu et al., *Strategies for Pre-training Graph Neural Networks*, ICLR 2020 — [arXiv:1905.12265](https://arxiv.org/abs/1905.12265)
- You et al., *Graph Contrastive Learning with Augmentations*, NeurIPS 2020 — [arXiv:2010.13902](https://arxiv.org/abs/2010.13902)
- Yang et al., *Analyzing Learned Molecular Representations for Property Prediction* (D-MPNN), JCIM 2019 — [doi](https://doi.org/10.1021/acs.jcim.9b00237)
- Sorkun et al., *AqSolDB*, Scientific Data 2019 — [링크](https://pmc.ncbi.nlm.nih.gov/articles/PMC6687799)
- Wu et al., *MoleculeNet*, Chemical Science 2018 — [doi](https://doi.org/10.1039/C7SC02664A)
- Gómez-Bombarelli et al., ACS Central Science 2018 (ZINC250k) — [doi](https://doi.org/10.1021/acscentsci.7b00572)
""")
