"""심화 분석: 사후 진단 · 라벨 신뢰도 · 불확실성 (AqSolDB scaffold_random)."""
import altair as alt
import pandas as pd
import streamlit as st
from common import BUDGETS, COLORS, LABELS, budget_label, read

st.title("심화 분석")
st.caption("모두 AqSolDB · scaffold_random · 시드 10개. 기술 통계이며 검정하지 않는다.")
X = alt.X("budget:O", sort=[budget_label(n) for n in BUDGETS], title="학습 라벨 수", axis=alt.Axis(labelAngle=0))

st.header("1. 사전학습 표현이 왜 나빴나", divider="gray")
st.markdown("D가 나쁘게 나온 **뒤에** 원인을 보려고 추가한 사후 분석이다 (사전 등록 아님, 판정은 바꾸지 않는다). "
            "인코더를 고정하고 분자 평균 표현 위에 Ridge 회귀만 학습했다.")
c1, c2 = st.columns([2, 3])
# 출처: reports/d_diagnosis_aqsoldb.md (scripts/diagnose_d.py). 인코더 임베딩이 필요해 앱에서 다시 계산하지 않는다
c1.dataframe(pd.DataFrame({"인코더": ["무작위 초기화", "D 마스킹", "D 대조학습"],
                           "원자 표현 평균 |h|": [0.074, 0.302, 0.048], "분자 표현 노름": [2.01, 10.85, 3.13]}),
             hide_index=True, width="stretch")
c2.dataframe(pd.DataFrame({"라벨 수": ["300", "1,000", "3,000", "전체"],
                           "무작위 초기화": [1.829, 1.591, 1.432, 1.420], "D 마스킹": [1.867, 1.729, 1.639, 1.603],
                           "D 대조학습": [1.936, 1.662, 1.537, 1.450]}).style.format(precision=3),
             hide_index=True, width="stretch")
st.markdown("""
- **선형 탐침 RMSE에서 무작위 초기화가 가장 좋다.** 사전학습이 용해도에 쓸모 있는 정보를 더하지 못하고 무관한 방향으로 표현을 정렬했다
- **마스킹 인코더는 출력 규모가 약 5배다.** 무작위 초기화 기준으로 잡은 미세조정 학습률·dropout 범위가 맞지 않았을 수 있다
- 가중치 로드·마스킹 누수 테스트와 사전학습 정확도(0.944 · 0.996)는 정상 → **버그가 아니라 부정적 전이**로 해석한다
""")

st.header("2. 라벨 신뢰도: 출처가 하나뿐인 분자", divider="gray")
st.markdown("AqSolDB는 분자마다 출처 수에 따른 신뢰 등급을 준다. test 분자를 **G1 단일 출처**(교차 검증 안 된 라벨, 약 712개)와 "
            "**G2~G5 복수 출처**(약 159개)로 나눴다.")
rel = read("reliability.csv")
r = rel.groupby(["model", "n_labels", "stratum"]).rmse.mean().reset_index()
r["budget"], r["name"] = r.n_labels.map(budget_label), r.model.map(LABELS)
models = [m for m in ["B_desc_lgbm", "D_contrast", "C_dmpnn_v2"] if m in set(r.model)]
names = [LABELS[m] for m in models]
base = alt.Chart(r).encode(
    x=X, y=alt.Y("rmse:Q", title="test RMSE (작을수록 좋음)", scale=alt.Scale(zero=False)),
    color=alt.Color("name:N", title="모델", sort=names, scale=alt.Scale(domain=names, range=[COLORS[m] for m in models]),
                    legend=alt.Legend(orient="top")),
    strokeDash=alt.StrokeDash("stratum:N", title="라벨 출처", legend=alt.Legend(orient="top")),
    tooltip=[alt.Tooltip("name:N", title="모델"), alt.Tooltip("stratum:N", title="층"),
             alt.Tooltip("budget:O", title="라벨 수"), alt.Tooltip("rmse:Q", title="RMSE", format=".3f")])
st.altair_chart((base.mark_line(strokeWidth=2) + base.mark_point(filled=True, size=60)).properties(height=360),
                width="stretch")
st.markdown("- **단일 출처 분자의 오차가 0.13~0.27 크다.** 세 모델 모두 같은 방향이다\n"
            "- 전체 예산에서 같은 층 안의 모델 간 차이는 0.04 이하지만, 같은 모델의 층 간 차이는 약 0.2다. "
            "이 데이터에서는 모델보다 **라벨 품질**이 더 큰 지렛대다")

st.header("3. 불확실성: 90% 예측 구간이 실제로 90%를 담는가", divider="gray")
st.markdown("주 그리드가 고른 HP를 그대로 써서 C v2와 D 대조학습에 세 방법을 적용했다. "
            "**deep ensemble**(시드 5개), **MC dropout**(30회), **split conformal**(valid 잔차의 90% 분위수).")
u = read("uncertainty.csv")
METHODS = {"cov_ens": "deep ensemble", "cov_mc": "MC dropout", "cov_conf": "split conformal"}
WIDTH = {"cov_ens": "width_ens", "cov_mc": "width_mc", "cov_conf": "width_conf"}
long = pd.concat([u.assign(method=v, coverage=u[k], width=u[WIDTH[k]]) for k, v in METHODS.items()])
long = long.groupby(["model", "n_labels", "method"])[["coverage", "width"]].mean().reset_index()
long["budget"], long["name"] = long.n_labels.map(budget_label), long.model.map(LABELS)
mnames = list(METHODS.values())
mcolor = alt.Color("method:N", title="방법", sort=mnames,
                   scale=alt.Scale(domain=mnames, range=["#2a78d6", "#eb6834", "#1baf7a"]), legend=alt.Legend(orient="top"))
tip = [alt.Tooltip("name:N", title="모델"), alt.Tooltip("method:N", title="방법"), alt.Tooltip("budget:O", title="라벨 수"),
       alt.Tooltip("coverage:Q", title="포함률", format=".2f"), alt.Tooltip("width:Q", title="평균 폭 (logS)", format=".2f")]
target = alt.Chart(pd.DataFrame({"y": [0.9]})).mark_rule(color="#7a7a80", strokeDash=[4, 3]).encode(y="y:Q")
for col, m in zip(st.columns(2), ["C_dmpnn_v2", "D_contrast"]):
    pts = alt.Chart(long[long.model == m]).encode(
        x=X, y=alt.Y("coverage:Q", title="실제 포함률", scale=alt.Scale(domain=[0, 1])), color=mcolor)
    layer = alt.layer(target, pts.mark_line(strokeWidth=2), pts.mark_point(filled=True, size=60).encode(tooltip=tip))
    col.altair_chart(layer.properties(height=300, title=LABELS[m]), width="stretch")
st.caption("점선이 목표 0.90. 점에 마우스를 올리면 평균 구간 폭이 나온다.")
st.markdown("""
- **앙상블·MC dropout은 크게 과신한다.** 90% 구간이 실제로는 21~43%만 담는다. 표준편차와 실제 오차의 순위 상관도 0.11 이하다.
  모델끼리의 의견 차이는 이 데이터 오차의 큰 몫인 라벨 노이즈와 새 골격으로의 이동을 담지 못한다
- **conformal은 거의 맞지만 넓다.** 포함률 0.83~0.88, 폭 3.0~4.5 logS. scaffold 분할에서는 valid와 test의 골격이 달라 목표에 조금 못 미친다
- 앙상블 평균은 단일 모델보다 RMSE를 0.005~0.043 줄인다
""")
