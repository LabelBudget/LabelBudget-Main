"""앱 페이지들이 함께 쓰는 경로·색·이름·데이터 로더. 표는 scripts/build_app_data.py가 만든다."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app" / "data"
REPO = "https://github.com/LabelBudget/LabelBudget-Main"

# 보고서 그림과 같은 색·순서 (analyze_baselines.COLORS, analyze_d.ORDER — 인접 쌍 색 구분 검증을 통과한 순서)
ORDER = ["B_desc_lgbm", "D_contrast", "C_dmpnn_v2", "D_mask", "C_fixed"]
COLORS = {"B_desc_lgbm": "#eb6834", "D_contrast": "#6250d6", "C_dmpnn_v2": "#eda100",
          "D_mask": "#e87ba4", "C_fixed": "#008300"}
LABELS = {"B_desc_lgbm": "B 서술자+LightGBM", "C_dmpnn_v2": "C v2 D-MPNN", "C_fixed": "C_fixed 크기 고정",
          "D_mask": "D 마스킹 사전학습", "D_contrast": "D 대조학습 사전학습"}
DATASETS = {"aqsoldb": "AqSolDB 용해도 (회귀)", "tox21_SR-ARE": "Tox21 SR-ARE 독성 (분류)"}
SPLITS = {"scaffold_random": "scaffold_random (주 분석)", "random": "random (낙관적 상한, H3)"}
METRIC = {"rmse": "RMSE (작을수록 좋음)", "auprc": "AUPRC (클수록 좋음)"}
BUDGETS = ["100", "300", "1000", "3000", "full"]


def budget_key(n: str) -> float:
    return float("inf") if n == "full" else float(n)


def budget_label(n: str) -> str:
    return "전체" if n == "full" else f"{int(n):,}"


@st.cache_data
def read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, dtype={"n_labels": str})


@st.cache_data
def read_json(name: str) -> dict:
    return json.loads((DATA / name).read_text())
