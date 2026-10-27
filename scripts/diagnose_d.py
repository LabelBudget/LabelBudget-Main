"""사후 진단 (사전 등록 아님): D의 사전학습 인코더가 왜 C_fixed보다 나빴는지.

python scripts/diagnose_d.py → reports/d_diagnosis_aqsoldb.md

D 결과(reports/d_aqsoldb.md)를 본 뒤에 추가한 분석이다. 판정을 바꾸지 않고 해석만 돕는다.
1. 인코더 출력 규모: 무작위 초기화 vs 사전학습 2종 (AqSolDB 전체 분자, 원자 표현과 분자 표현)
2. 선형 탐침: 인코더를 고정하고 분자 평균 표현 위에 Ridge만 학습한다 (scaffold_random 시드 10개, alpha는 valid로 선택).
   무작위 초기화 인코더의 표현도 꽤 강한 기준선이라는 점(무작위 특징)을 함께 본다
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge

from labelbudget.evaluation.budget import subsample_train
from labelbudget.featurize.graph import batch_graphs, mol_graph
from labelbudget.models.dmpnn import DMPNNEncoder, readout
from labelbudget.models.gnn_space import FIXED_BACKBONE
from labelbudget.splits.strategies import make_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
ENCODERS = ["무작위 초기화", "D_mask", "D_contrast"]


def embed(G, name: str) -> tuple[np.ndarray, dict]:
    torch.manual_seed(0)  # 무작위 초기화는 C_fixed가 쓰는 것과 같은 분포의 한 표본
    enc = DMPNNEncoder(**FIXED_BACKBONE).eval()
    if name != "무작위 초기화":
        enc.load_state_dict(torch.load(ROOT / "models" / "pretrained" / f"{name}.pt"))
    mean, atom_abs, n_atoms = [], 0.0, 0
    with torch.no_grad():
        for i in range(0, len(G), 512):
            bg = batch_graphs(G[i:i + 512])
            h = enc(bg)
            atom_abs += float(h.abs().sum())
            n_atoms += h.numel()
            mean.append(readout(h, bg.batch, bg.n_mols, "mean").numpy())
    E = np.concatenate(mean)
    return E, {"원자 표현 평균 |h|": atom_abs / n_atoms, "분자 평균 표현 노름": float(np.linalg.norm(E, axis=1).mean()),
               "한 번도 켜지지 않은 차원": int((np.abs(E).max(0) == 0).sum())}


def main() -> int:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    conf = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    df = pd.read_csv(ROOT / "data" / "processed" / "aqsoldb.csv")
    y = df.label.to_numpy()
    G = [mol_graph(s) for s in df.smiles]
    embs, scale = {}, {}
    for name in ENCODERS:
        embs[name], scale[name] = embed(G, name)
    rows = []
    for seed in conf["seeds"]["split_seeds"]:
        sp = make_split(df, strategy="scaffold_random", seed=seed, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        for n in [100, 300, 1000, 3000, None]:
            tr = sp.train[subsample_train(y[sp.train], n_labels=n, seed=seed + conf["seeds"]["subsample_seed_offset"],
                                          task="regression")]
            for name, E in embs.items():
                Z = (E - E[tr].mean(0)) / (E[tr].std(0) + 1e-8)
                rmse = lambda m, idx: float(np.sqrt(((m.predict(Z[idx]) - y[idx]) ** 2).mean()))  # noqa: E731
                a = min(ALPHAS, key=lambda a: rmse(Ridge(alpha=a).fit(Z[tr], y[tr]), sp.valid))
                rows.append({"seed": seed, "n": "full" if n is None else str(n), "enc": name,
                             "rmse": rmse(Ridge(alpha=a).fit(Z[tr], y[tr]), sp.test)})
    r = pd.DataFrame(rows)
    t = r.pivot_table(index="n", columns="enc", values="rmse", aggfunc="mean")
    t = t.reindex(["100", "300", "1000", "3000", "full"])[ENCODERS]
    w = r.pivot_table(index=["n", "seed"], columns="enc", values="rmse")
    md = ["# 사후 진단: 사전학습 인코더 — AqSolDB", "",
          "> **사전 등록 분석이 아니다.** `reports/d_aqsoldb.md`에서 D가 C_fixed보다 나쁘게 나온 뒤 원인을 보려고 추가했다. 판정은 바꾸지 않는다.", "",
          "## 1. 인코더 출력 규모 (AqSolDB 8,703개 분자)", "",
          "| 인코더 | " + " | ".join(scale[ENCODERS[0]]) + " |", "|---|" + "---:|" * len(scale[ENCODERS[0]])]
    md += [f"| {e} | " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in scale[e].values()) + " |"
           for e in ENCODERS]
    md += ["", "## 2. 선형 탐침 (인코더 고정 + Ridge, test RMSE 시드 10개 평균)", "",
           "| 라벨 수 | " + " | ".join(ENCODERS) + " | D_mask가 무작위보다 나은 시드 | D_contrast가 무작위보다 나은 시드 |",
           "|---:|" + "---:|" * (len(ENCODERS) + 2)]
    for n in t.index:
        x = w.loc[n]
        md.append(f"| {n} | " + " | ".join(f"{t.loc[n, e]:.3f}" for e in ENCODERS)
                  + f" | {int((x['D_mask'] < x['무작위 초기화']).sum())}/10 | {int((x['D_contrast'] < x['무작위 초기화']).sum())}/10 |")
    md += ["", "라벨 100개의 Ridge는 300차원에 학습 100개라 불안정하다 (선형계 경고). 이 행은 참고만 한다.", ""]
    out = ROOT / "reports" / "d_diagnosis_aqsoldb.md"
    out.write_text("\n".join(md))
    print(f"→ {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
