"""라벨 예산 곡선과 사전 등록 비교 (시드별)."""
import altair as alt
import pandas as pd
import streamlit as st
from common import BUDGETS, COLORS, DATASETS, LABELS, METRIC, ORDER, SPLITS, budget_label, read

st.title("라벨 예산 곡선")
st.caption("값은 시드 10개의 test 성능. 탐색 횟수는 라벨 ≤1,000개 20회, ≥3,000개 10회 체크포인트이며 모든 모델이 같다.")

curves, comps = read("curves.csv"), read("comparisons.csv")
c1, c2 = st.columns(2)
ds = c1.selectbox("데이터셋", list(DATASETS), format_func=DATASETS.get)
split = c2.selectbox("분할", list(SPLITS), format_func=SPLITS.get)
d = curves[(curves.dataset == ds) & (curves.split == split)]
metric = d.metric.iloc[0]
available = [m for m in ORDER if m in set(d.model)]
models = st.multiselect("모델", available, default=available, format_func=LABELS.get)
if split == "random":
    st.caption("random 분할(H3)은 C v2와 valid 점수로 고른 D 하나만 돌렸다. 같은 골격이 학습·test에 함께 들어가 성능이 낙관적이다.")

d = d[d.model.isin(models)]
n_train = d.groupby("n_labels").n_train.mean()
xlab = {n: (f"전체 ({n_train[n]:,.0f})" if n == "full" else budget_label(n)) for n in n_train.index}
budgets = [n for n in BUDGETS if n in n_train.index]
agg = (d.groupby(["model", "n_labels"]).metric_value.agg(["mean", "std", "count"]).reset_index()
       .assign(lo=lambda x: x["mean"] - x["std"], hi=lambda x: x["mean"] + x["std"],
               budget=lambda x: x.n_labels.map(xlab), name=lambda x: x.model.map(LABELS)))
order = [xlab[n] for n in budgets]
names = [LABELS[m] for m in ORDER if m in models]
color = alt.Color("name:N", title="모델", sort=names,
                  scale=alt.Scale(domain=names, range=[COLORS[m] for m in ORDER if m in models]),
                  legend=alt.Legend(orient="top", symbolOpacity=1, symbolType="circle"))
x = alt.X("budget:O", sort=order, title="학습 라벨 수", axis=alt.Axis(labelAngle=0))
tip = [alt.Tooltip("name:N", title="모델"), alt.Tooltip("budget:O", title="라벨 수"),
       alt.Tooltip("mean:Q", title="평균", format=".3f"), alt.Tooltip("std:Q", title="SD", format=".3f"),
       alt.Tooltip("count:Q", title="시드")]
base = alt.Chart(agg).encode(x=x, color=color)
chart = (base.mark_area(opacity=0.12).encode(y=alt.Y("lo:Q", title=f"test {METRIC[metric]}", scale=alt.Scale(zero=False)),
                                            y2="hi:Q")
         + base.mark_line(strokeWidth=2).encode(y="mean:Q")
         + base.mark_point(filled=True, size=70, stroke="white", strokeWidth=1.5, opacity=1).encode(y="mean:Q", tooltip=tip))
st.altair_chart(chart.properties(height=420), width="stretch")
st.caption("선은 시드 10개 평균, 띠는 ±1 SD. 점에 마우스를 올리면 값이 나온다.")

with st.expander("표로 보기"):
    t = agg.pivot(index="n_labels", columns="model", values="mean").reindex(budgets)[[m for m in ORDER if m in models]]
    st.dataframe(t.rename(index=xlab, columns=LABELS).style.format("{:.3f}"), width="stretch")

st.subheader("사전 등록 비교 — 시드별 차이")
cmp = comps[(comps.dataset == ds) & (comps.split == split)]
pairs = list(dict.fromkeys(zip(cmp.a, cmp.b)))
pair = st.selectbox("비교", pairs, format_func=lambda p: f"{LABELS[p[0]]} − {LABELS[p[1]]}")
a, b = pair
higher = metric == "auprc"
w = curves[(curves.dataset == ds) & (curves.split == split) & curves.model.isin([a, b])].pivot_table(
    index=["n_labels", "split_seed"], columns="model", values="metric_value").dropna().reset_index()
w["diff"] = w[a] - w[b]
w["budget"] = w.n_labels.map(xlab)
w["better"] = (w["diff"] > 0) if higher else (w["diff"] < 0)
w["who"] = w.better.map({True: f"{LABELS[a]} 우세", False: f"{LABELS[b]} 우세"})
dots = alt.Chart(w).mark_circle(size=80, opacity=0.9, stroke="white", strokeWidth=1).encode(
    x=alt.X("budget:O", sort=order, title="학습 라벨 수", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("diff:Q", title=f"{LABELS[a]} − {LABELS[b]} ({metric.upper()})"),
    color=alt.Color("who:N", title=None, scale=alt.Scale(domain=[f"{LABELS[a]} 우세", f"{LABELS[b]} 우세"],
                                                         range=[COLORS[a], COLORS[b]]), legend=alt.Legend(orient="top")),
    tooltip=[alt.Tooltip("split_seed:Q", title="시드"), alt.Tooltip("diff:Q", title="차이", format="+.3f")])
zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#7a7a80", strokeDash=[4, 3]).encode(y="y:Q")
st.altair_chart((zero + dots).properties(height=320), width="stretch")
better = "클수록" if higher else "작을수록"
st.caption(f"점 하나가 시드 하나. {metric.upper()}는 {better} 좋으므로 "
           f"{'0보다 위' if higher else '0보다 아래'}가 {LABELS[a]} 우세다.")

r = cmp[(cmp.a == a) & (cmp.b == b)].copy()
r["order"] = r.n_labels.map(lambda n: budgets.index(n) if n in budgets else 99)
r = r.sort_values("order")
st.dataframe(pd.DataFrame({
    "라벨 수": r.n_labels.map(xlab).to_numpy(), "중앙값 차이": r.median_diff.map("{:+.3f}".format).to_numpy(),
    f"{LABELS[a]} 우세 시드": [f"{k}/{n}" for k, n in zip(r.a_better, r.pairs)],
    "p": r.p.map("{:.4f}".format).to_numpy(), "p (Holm)": r.p_holm.map("{:.4f}".format).to_numpy(),
    "판정 (Holm < 0.05)": ["유의" if p < 0.05 else "—" for p in r.p_holm]}), hide_index=True, width="stretch")
st.caption("시드 쌍 Wilcoxon 부호순위 검정, 비교마다 예산 간 Holm 보정. 판정 규칙은 결과 전에 고정했다 (05_PRETRAIN_SPEC.md §4).")
