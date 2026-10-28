"""데모 앱(결과 탐색기)이 읽는 작은 표를 만든다. 앱은 분석 코드·scipy 없이 이 파일들만 읽는다.

python scripts/build_app_data.py → app/data/*.csv

- curves.csv       데이터셋·분할·모델·예산·시드별 주 지표 (예산별 체크포인트, 05 §2.2 T1)
- comparisons.csv  사전 등록 비교 (시드 쌍 Wilcoxon, 예산 간 Holm). analyze_d.compare 그대로
- uncertainty.csv  조합별 포함률·폭·순위 상관 (analyze_uncertainty.combo_stats 그대로)
- reliability.csv  모델·예산·시드·층별 RMSE (analyze_reliability와 같은 계산)
- train.csv        앱 모델 학습 분자 (scaffold_random 시드 0, 전체) + 실측 logS + 이름. 최근접 이웃 표시용
- examples.csv     예시 분자: 앱 학습셋 밖(valid·test)과 안, 적용 범위 밖, 거부 사례
- conformal.json   앱 D 모델(시드 0, 전체 예산)의 conformal 반폭 q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_baselines import PRIMARY, budget_order  # noqa: E402
from analyze_d import TAGS, at_checkpoint, compare, load, trials_for  # noqa: E402
from analyze_uncertainty import combo_stats  # noqa: E402

from labelbudget.evaluation.budget import subsample_train  # noqa: E402
from labelbudget.splits.strategies import make_split  # noqa: E402

RES = ROOT / "reports" / "results"
OUT = ROOT / "app" / "data"
DATASETS = ["aqsoldb", "tox21_SR-ARE"]
SPLITS = ["scaffold_random", "random"]
# 이름은 AqSolDB 원본 Name 그대로. 일상에서 알려진 분자를 고르고, 앱 학습셋 안/밖을 함께 표시한다
EXAMPLE_NAMES = ["caffeine", "naphthalene", "benzene", "ethanol", "toluene", "anthracene", "pyrene", "urea",
                 "testosterone", "Lindane", "phenol", "benzoic acid", "p,p'-DDT", "phenytoin", "chlorobenzene", "carbamazepine", "estradiol", "camphor", "hexane",
                 "acetone", "Cholesterol"]
EDGE_CASES = [  # (이름, SMILES, 무엇을 보여 주나)
    ("아스피린", "CC(=O)Oc1ccccc1C(=O)O", "기본값. 학습셋에 있는 분자"),
    ("붕소 케이지", "B12B3B4B1C234", "적용 범위 밖 경고 (GNN이 −65로 외삽했던 분자, LIMITATIONS)"),
    ("페로센", "[CH-]1C=CC=C1.[CH-]1C=CC=C1.[Fe+2]", "거부: 다성분 (금속 착물)"),
    ("염화나트륨", "[Na+].[Cl-]", "거부: 다성분 (염)"),
    ("메탄", "C", "거부: 무거운 원자 3개 미만"),
]


def build_results() -> None:
    curves, comps = [], []
    for ds in DATASETS:
        for split in SPLITS:
            allk, models = load(ds, split)
            metric, higher = PRIMARY[allk.task.iloc[0]]
            d = at_checkpoint(allk)
            r = d[d.metric_name == metric]
            curves.append(r[["model", "n_labels", "split_seed", "metric_value", "n_train"]]
                          .assign(dataset=ds, split=split, metric=metric))
            w = r.pivot_table(index=["n_labels", "split_seed"], columns="model", values="metric_value")
            pairs = [(dm, ref) for dm in ("D_mask", "D_contrast") if dm in models
                     for ref in ("C_fixed", "C_dmpnn_v2", "B_desc_lgbm") if ref in models]
            pairs += [(c, ref) for c, ref in (("C_dmpnn_v2", "B_desc_lgbm"), ("C_dmpnn_v2", "C_fixed"))
                      if c in models and ref in models]
            for a, b in pairs:
                for x in compare(w, a, b, higher):
                    comps.append({"dataset": ds, "split": split, "a": a, "b": b, "metric": metric, "higher": higher,
                                  "n_labels": x["n_labels"], "pairs": x["n"], "median_diff": x["median_diff"],
                                  "a_better": x["a_better"], "p": x["p_value"], "p_holm": x["p_holm"]})
    pd.concat(curves).to_csv(OUT / "curves.csv", index=False)
    pd.DataFrame(comps).to_csv(OUT / "comparisons.csv", index=False)


def build_uncertainty() -> None:
    rows = []
    for f in sorted((RES / "uncertainty").glob("*.csv")):
        model, s, n = f.stem.rsplit("_", 2)
        rows.append({"model": model, "seed": int(s[1:]), "n_labels": n[1:], **combo_stats(pd.read_csv(f))})
    pd.DataFrame(rows).drop(columns="dropout0").to_csv(OUT / "uncertainty.csv", index=False)
    q = pd.read_csv(RES / "uncertainty" / "D_contrast_s0_nfull.csv").conformal_q.iloc[0]
    (OUT / "conformal.json").write_text(json.dumps({"D_contrast": float(q), "coverage_target": 0.9,
                                                    "source": "reports/results/uncertainty/D_contrast_s0_nfull.csv"}))


def build_reliability() -> None:
    grp = pd.read_csv(ROOT / "data" / "processed" / "aqsoldb.csv", usecols=["inchikey", "Group"]).set_index("inchikey").Group
    dirs = {"B_desc_lgbm": RES / "preds" / "aqsoldb_baselines_rerun", "C_dmpnn_v2": RES / "preds" / "aqsoldb_gnn_v2_full",
            "D_contrast": RES / "preds" / f"aqsoldb_gnn_{TAGS['D_contrast']}_full"}
    rows = []
    for model, d in dirs.items():
        for f in sorted(d.glob(f"{model}_scaffold_random_s*_n*.csv")):
            seed, n = f.stem.split("_s")[-1].split("_n")
            x = pd.read_csv(f)
            x["stratum"] = np.where(x.inchikey.map(grp) == "G1", "G1 단일 출처", "G2~G5 복수 출처")
            for s, g in x.groupby("stratum"):
                rows.append({"model": model, "seed": int(seed), "n_labels": n, "stratum": s, "n": len(g),
                             "rmse": float(np.sqrt(((g.y_true - g[f"pred_at_{trials_for(n)}"]) ** 2).mean()))})
    pd.DataFrame(rows).to_csv(OUT / "reliability.csv", index=False)


def build_molecules() -> None:
    conf = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    df = pd.read_csv(ROOT / "data" / "processed" / "aqsoldb.csv")
    raw = pd.read_csv(ROOT / "data" / "raw" / "aqsoldb.csv", usecols=["Name", "InChIKey"])
    name = raw.drop_duplicates("InChIKey").set_index("InChIKey").Name
    df["name"] = df.inchikey.map(name)
    sp = conf["splits"]
    split = make_split(df, strategy="scaffold_random", seed=0, frac_train=sp["frac_train"],
                       frac_valid=sp["frac_valid"], frac_test=sp["frac_test"])
    y = df.label.to_numpy()
    tr = split.train[subsample_train(y[split.train], n_labels=None, seed=conf["seeds"]["subsample_seed_offset"],
                                     task="regression")]
    app_train = (ROOT / "models" / "app" / "train_smiles.txt").read_text().splitlines()
    if df.smiles.to_numpy()[tr].tolist() != app_train:
        raise RuntimeError("앱 모델 학습셋과 다시 만든 학습셋이 다르다")
    df.iloc[tr][["smiles", "label", "name"]].to_csv(OUT / "train.csv", index=False)

    role = pd.Series("학습", index=df.index)
    role.iloc[split.valid] = "valid"
    role.iloc[split.test] = "test"
    df["role"] = role
    rows = []
    for n in EXAMPLE_NAMES:
        hit = df[df.name.str.lower() == n.lower()]
        if len(hit) != 1:
            raise RuntimeError(f"예시 분자 {n}: {len(hit)}개 일치")
        h = hit.iloc[0]
        rows.append({"name": h["name"], "smiles": h.smiles, "measured": h.label, "role": h.role,
                     "note": "앱 학습셋 밖 — 실측값과 비교해 볼 수 있다" if h.role != "학습" else "앱 학습셋 안"})
    rows += [{"name": n, "smiles": s, "measured": np.nan, "role": "—", "note": note} for n, s, note in EDGE_CASES]
    pd.DataFrame(rows).to_csv(OUT / "examples.csv", index=False)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    build_results()
    build_uncertainty()
    build_reliability()
    build_molecules()
    for f in sorted(OUT.iterdir()):
        print(f"→ {f.relative_to(ROOT)} ({f.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
