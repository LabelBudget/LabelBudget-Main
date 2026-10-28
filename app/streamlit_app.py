"""LabelBudget 데모 앱: 연구 결과 탐색 + 수용해도(logS) 예측 (05 §2.2 U3).

streamlit run app/streamlit_app.py      (앱 전용 환경: streamlit은 본 환경에 넣지 않는다)

- 결과 페이지는 app/data/*.csv만 읽는다 (scripts/build_app_data.py가 만든다)
- 예측 페이지의 모델은 AqSolDB scaffold_random 시드 0, 전체 학습셋(6,962개)으로 학습했다 (scripts/train_app_models.py)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# libomp 충돌 (torch와 LightGBM이 각자 libomp를 싣는다): torch를 먼저 실으면 이후 OpenMP 연산이 세그폴트로 죽는다
# (2026-10-27 실측). 모든 페이지가 한 프로세스에서 돌므로 여기서 LightGBM을 먼저 싣고, torch는 1스레드로만 쓴다
import lightgbm  # noqa: E402, F401
import torch  # noqa: E402

torch.set_num_threads(1)
import streamlit as st  # noqa: E402

st.set_page_config(page_title="LabelBudget", page_icon="🧪", layout="wide")
VIEWS = Path(__file__).resolve().parent / "views"
st.navigation([
    st.Page(VIEWS / "overview.py", title="개요", icon="🏠", default=True),
    st.Page(VIEWS / "curves.py", title="라벨 예산 곡선", icon="📈"),
    st.Page(VIEWS / "analysis.py", title="심화 분석", icon="🔬"),
    st.Page(VIEWS / "predict.py", title="용해도 예측", icon="🧪"),
]).run()
