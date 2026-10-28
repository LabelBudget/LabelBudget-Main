"""LabelBudget 데모: 수용해도(logS) 예측 (05 §2.2 U3). 연구 결과가 아니라 시연용이다.

streamlit run app/streamlit_app.py      (앱 전용 환경: streamlit은 본 환경에 넣지 않는다)

- B(서술자+LightGBM)와 "더 나은 D"(사전학습 D-MPNN)의 예측을 나란히 보여 준다
- 모델은 AqSolDB scaffold_random 시드 0, 전체 학습셋(6,962개)으로 학습했다 (scripts/train_app_models.py)
- 학습 분자와의 최근접 ECFP4 Tanimoto가 0.4 미만이면 적용 범위 밖으로 경고한다
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# libomp 충돌 (torch와 LightGBM이 각자 libomp를 싣는다): torch를 먼저 실으면 이후 OpenMP 연산이 세그폴트로 죽는다
# (2026-10-27 실측). LightGBM을 먼저 싣고, torch는 1스레드로만 쓰고, LightGBM 예측도 1스레드로 한다
import lightgbm as lgb  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402
from rdkit import Chem, DataStructs  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

from labelbudget.curate.standardize import StandardizeConfig, Standardizer, StereoPolicy  # noqa: E402
from labelbudget.featurize.descriptors import compute_descriptors  # noqa: E402
from labelbudget.featurize.graph import batch_graphs, mol_graph  # noqa: E402
from labelbudget.models.dmpnn import DMPNN  # noqa: E402

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
    train = (APP / "train_smiles.txt").read_text().splitlines()
    fps = [_fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train]
    conf = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())["standardize"]
    std = Standardizer(StandardizeConfig(**{**conf, "stereo": StereoPolicy(conf["stereo"])}))
    return meta, d_name, booster, cols, d, ck["mu"], ck["sigma"], train, fps, std


def main() -> None:
    st.set_page_config(page_title="LabelBudget 용해도 데모", layout="centered")
    meta, d_name, booster, cols, d, mu, sigma, train, fps, std = load()
    st.title("수용해도(logS) 예측 데모")
    st.caption("LabelBudget 연구의 시연용 앱입니다. 실험값을 대신하지 않으며 의사결정에 쓰지 마세요.")
    smi = st.text_input("SMILES", value="CC(=O)Oc1ccccc1C(=O)O")
    if not smi:
        return
    r = std(smi)
    if not r.ok:
        st.error(f"표준화 실패: {r.reason}. 연구 데이터와 같은 규칙(단일 성분 유기 분자, 무거운 원자 3~100개)만 받습니다.")
        return
    x = compute_descriptors([r.smiles])[cols].to_numpy(dtype=np.float32)
    b_pred = float(booster.predict(x, num_threads=1)[0])
    with torch.no_grad():
        d_pred = float(d(batch_graphs([mol_graph(r.smiles)])).item() * sigma + mu)
    sims = DataStructs.BulkTanimotoSimilarity(_fpgen.GetFingerprint(Chem.MolFromSmiles(r.smiles)), fps)
    k = int(np.argmax(sims))

    c1, c2 = st.columns(2)
    c1.metric("B  서술자 + LightGBM", f"{b_pred:.2f} logS")
    c2.metric(f"D  {'마스킹' if d_name == 'D_mask' else '대조학습'} 사전학습 D-MPNN", f"{d_pred:.2f} logS")
    st.write(f"표준화된 SMILES: `{r.smiles}`")
    if sims[k] < AD_THRESHOLD:
        st.warning(f"적용 범위 밖일 수 있습니다: 학습 분자와의 최대 Tanimoto 유사도 {sims[k]:.2f} (< {AD_THRESHOLD}).")
    else:
        st.info(f"가장 비슷한 학습 분자 (Tanimoto {sims[k]:.2f}): `{train[k]}`")
    with st.expander("모델 정보"):
        # 모델을 행으로 둔다 (열마다 한 가지 타입이어야 Arrow 변환이 된다)
        st.dataframe(pd.DataFrame.from_dict({m: {"학습 분자": v["n_train"], "test RMSE (시드 0)": round(v["test_rmse"], 3),
                                                 "분할": v["split"]} for m, v in meta.items()}, orient="index"))
        st.markdown("연구 결과 전체는 `reports/`의 보고서를 참고하세요. 시드 10개 평균과 통계 검정이 그쪽에 있습니다.")


main()
