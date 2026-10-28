"""수용해도(logS) 예측: B(서술자+LightGBM)와 D(대조학습 사전학습 D-MPNN)를 나란히. 시연용이다.

- 모델: AqSolDB scaffold_random 시드 0, 전체 학습셋 6,962개 (scripts/train_app_models.py, test RMSE가 주 그리드와 같음)
- D의 90% 구간: 같은 모델의 split conformal 반폭 q (모든 분자에 같은 폭). 같은 설정의 실제 포함률은 0.88
- 학습 분자와의 최근접 ECFP4 Tanimoto가 0.4 미만이면 적용 범위 밖으로 경고한다 (관행적 값, 검증하지 않음)
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from common import DATA, ROOT, read, read_json
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from labelbudget.curate.standardize import StandardizeConfig, Standardizer, StereoPolicy
from labelbudget.featurize.descriptors import compute_descriptors
from labelbudget.featurize.graph import batch_graphs, mol_graph
from labelbudget.models.dmpnn import DMPNN

APP = ROOT / "models" / "app"
AD_THRESHOLD = 0.4
_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@st.cache_resource
def load():
    meta = json.loads((APP / "meta.json").read_text())
    d_name = next(k for k in meta if k.startswith("D_"))
    booster = lgb.Booster(model_file=str(APP / "B_desc_lgbm.txt"))
    cols = json.loads((APP / "B_desc_lgbm_columns.json").read_text())
    ck = torch.load(APP / f"{d_name}.pt", map_location="cpu")
    p = ck["params"]
    d = DMPNN(d_h=p["d_h"], depth=p["depth"], dropout=p["dropout"], ffn_hidden=p["ffn_hidden"],
              ffn_layers=p["ffn_layers"], readout=p.get("readout", "mean"))
    d.load_state_dict(ck["state_dict"])
    d.eval()
    train = pd.read_csv(DATA / "train.csv")
    fps = [_fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.smiles]
    conf = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["standardize"]
    std = Standardizer(StandardizeConfig(**{**conf, "stereo": StereoPolicy(conf["stereo"])}))
    return meta, d_name, booster, cols, d, ck["mu"], ck["sigma"], train, fps, std


def draw(smiles: str) -> str | None:
    # 그림은 부가 기능이다. 서버 환경에서 그리기가 실패해도 예측은 보여 준다
    # (Linux RDKit 휠의 cairo는 libX11·libXext·libXrender·libexpat을 시스템에서 찾는다 → packages.txt)
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        dr = rdMolDraw2D.MolDraw2DSVG(360, 240)
        dr.drawOptions().clearBackground = False
        dr.DrawMolecule(Chem.MolFromSmiles(smiles))
        dr.FinishDrawing()
        return dr.GetDrawingText()
    except Exception:
        return None


meta, d_name, booster, cols, model_d, mu, sigma, train, fps, std = load()
examples = read("examples.csv")
q = read_json("conformal.json")[d_name]

st.title("수용해도(logS) 예측")
st.caption("LabelBudget 연구의 시연용 앱입니다. 실험값을 대신하지 않으며 의사결정에 쓰지 마세요. "
           "연구 결론대로, 사전학습 모델(D)이 서술자 모델(B)보다 낫지 않습니다. 두 모델이 어떻게 다르게 예측하는지 보는 용도입니다.")

label = {r.name: f"{r.name} — {r.note}" for r in examples.itertuples()}
pick = st.selectbox("예시 분자", ["직접 입력", *examples.name], index=1 + int(np.flatnonzero(examples.name == "아스피린")[0]),
                    format_func=lambda n: label.get(n, n))
default = "" if pick == "직접 입력" else examples.set_index("name").smiles[pick]
smi = st.text_input("SMILES", value=default, key=f"smiles_{pick}", placeholder="예: CC(=O)Oc1ccccc1C(=O)O")
if not smi:
    st.stop()
r = std(smi)
if not r.ok:
    st.error(f"표준화 실패: {r.reason}. 연구 데이터와 같은 규칙(단일 성분 유기 분자, 무거운 원자 3~100개)만 받습니다.")
    st.stop()

x = compute_descriptors([r.smiles])[cols].to_numpy(dtype=np.float32)
b_pred = float(booster.predict(x, num_threads=1)[0])
with torch.no_grad():
    d_pred = float(model_d(batch_graphs([mol_graph(r.smiles)])).item() * sigma + mu)
sims = np.array(DataStructs.BulkTanimotoSimilarity(_fpgen.GetFingerprint(Chem.MolFromSmiles(r.smiles)), fps))
near = np.argsort(-sims)[:3]

left, right = st.columns([3, 2])
with left:
    c1, c2 = st.columns(2)
    c1.metric("B  서술자 + LightGBM", f"{b_pred:.2f} logS")
    c2.metric(f"D  {'마스킹' if d_name == 'D_mask' else '대조학습'} 사전학습 D-MPNN", f"{d_pred:.2f} logS")
    c2.caption(f"90% 구간 {d_pred - q:.1f} ~ {d_pred + q:.1f}", help="split conformal. 아래 \"모델 정보와 구간의 의미\" 참고")
    known = train[train.smiles == r.smiles]
    ex = examples[(examples.smiles == r.smiles) & examples.measured.notna()]
    if len(known):
        st.write(f"**실측 logS {known.label.iloc[0]:.2f}** — 학습에 쓴 분자라 정답을 이미 본 셈이다.")
    elif len(ex):
        y = ex.measured.iloc[0]
        st.write(f"**실측 logS {y:.2f}** ({ex.role.iloc[0]} 분자, 학습에 쓰지 않음) → "
                 f"오차 B {b_pred - y:+.2f}, D {d_pred - y:+.2f}")
    if sims[near[0]] < AD_THRESHOLD:
        st.warning(f"적용 범위 밖일 수 있습니다: 학습 분자와의 최대 Tanimoto 유사도 {sims[near[0]]:.2f} (< {AD_THRESHOLD}). "
                   "비슷한 분자를 본 적이 없어 예측을 믿기 어렵습니다.")
    st.caption(f"표준화된 SMILES: `{r.smiles}`")
with right:
    svg = draw(r.smiles)
    if svg:
        st.image(svg, width="stretch")

st.subheader("가장 비슷한 학습 분자")
st.dataframe(pd.DataFrame({"이름": train.name.iloc[near].fillna("—").to_numpy(), "SMILES": train.smiles.iloc[near].to_numpy(),
                           "실측 logS": train.label.iloc[near].round(2).to_numpy(), "Tanimoto": sims[near].round(2)}),
             hide_index=True, width="stretch")
st.caption("모델 예측이 이웃 분자의 실측값과 크게 다르면 근거를 의심해 볼 만하다. ECFP4(반지름 2, 2048비트) 기준.")

with st.expander("모델 정보와 구간의 의미"):
    # 모델을 행으로 둔다 (열마다 한 가지 타입이어야 Arrow 변환이 된다)
    st.dataframe(pd.DataFrame.from_dict({m: {"학습 분자": v["n_train"], "test RMSE (시드 0)": round(v["test_rmse"], 3),
                                             "분할": v["split"]} for m, v in meta.items()}, orient="index"))
    st.markdown(f"""
- **D의 90% 구간**은 split conformal이다: valid 분자의 절대 오차 90% 분위수 q = {q:.2f}를 예측에 더하고 뺀다.
  모든 분자에 같은 폭이고, 같은 설정에서 test 실제 포함률은 0.88이었다 (새 골격이라 목표 0.90보다 조금 낮다)
- 앙상블·MC dropout 구간은 실제 포함률이 21~43%라 보여 주지 않는다 (심화 분석 페이지)
- 오차 감각: test RMSE 약 1.0 logS, 측정 노이즈 하한 약 0.3 logS (출처 간 표준편차)
""")
