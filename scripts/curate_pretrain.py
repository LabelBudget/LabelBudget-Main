"""사전학습용 ZINC250k 정리. 스펙 05 §2 P1·P2, §3.

python scripts/curate_pretrain.py [--workers 2]

1. data/raw/zinc250k.csv (없으면 내려받음) SHA-256 확인
2. 평가 데이터와 같은 표준화 (configs standardize). 거부·중복(InChIKey) 제거
3. P2: 평가 데이터셋(AqSolDB·Tox21 SR-ARE·ESOL) 중 하나와 InChIKey 골격(첫 블록)이 같은 분자 제거
   → data/processed/zinc250k_pretrain.csv + _ledger.json, docs/data_card_zinc250k_pretrain.md
"""
import argparse
import hashlib
import json
import urllib.request
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml

from labelbudget.curate.standardize import StandardizeConfig, Standardizer, StereoPolicy

ROOT = Path(__file__).resolve().parents[1]
URL = ("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/"
       "models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")
SHA256 = "35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae"
EVAL_SETS = ["aqsoldb", "tox21_SR-ARE", "esol"]

_std: Standardizer | None = None


def _init(conf: dict) -> None:
    global _std
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _std = Standardizer(StandardizeConfig(**{**conf, "stereo": StereoPolicy(conf["stereo"])}))


def _one(smiles: str) -> tuple[str | None, str | None, str | None]:
    r = _std(smiles)
    return (r.smiles, r.inchikey, None) if r.ok else (None, None, r.reason)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    raw = ROOT / "data" / "raw" / "zinc250k.csv"
    if not raw.exists():
        urllib.request.urlretrieve(URL, raw)
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    if sha != SHA256:
        raise RuntimeError(f"ZINC250k 해시 불일치: {sha}")
    smiles = pd.read_csv(raw).smiles.str.strip().tolist()  # 원본 SMILES 끝에 줄바꿈이 붙어 있다

    conf = yaml.safe_load(args.config.read_text())["standardize"]
    with Pool(args.workers, initializer=_init, initargs=(conf,)) as pool:
        res = pool.map(_one, smiles, chunksize=500)
    df = pd.DataFrame(res, columns=["smiles", "inchikey", "reason"])
    rejected = Counter(df.reason.dropna())
    df = df[df.reason.isna()].drop(columns="reason")
    n_std = len(df)
    df = df.drop_duplicates("inchikey")
    n_dedup = len(df)

    eval_skel = {}
    for name in EVAL_SETS:
        keys = pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv", usecols=["inchikey"]).inchikey
        eval_skel[name] = set(keys.str.split("-").str[0])
    skel = df.inchikey.str.split("-").str[0]
    hit = {name: int(skel.isin(s).sum()) for name, s in eval_skel.items()}
    drop = skel.isin(set().union(*eval_skel.values()))
    df = df[~drop]

    out = ROOT / "data" / "processed" / "zinc250k_pretrain.csv"
    df.to_csv(out, index=False)
    ledger = {"source": URL, "sha256": sha, "n_raw": len(smiles), "rejected": dict(rejected),
              "n_after_standardize": n_std, "n_duplicates": n_std - n_dedup, "n_after_dedupe": n_dedup,
              "skeleton_overlap_by_eval_set": hit, "n_removed_overlap": int(drop.sum()), "n_final": len(df),
              "standardize": conf}
    (out.parent / "zinc250k_pretrain_ledger.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
    card = [
        "# 데이터 카드: ZINC250k 사전학습 세트",
        "",
        f"- 출처: `{URL}`",
        f"- SHA-256: `{sha}`",
        "- 용도: 자기지도 사전학습 전용 (라벨 없음). 스펙 05 §2 P1·P2",
        "- 생성: `python scripts/curate_pretrain.py`",
        "",
        "| 단계 | 분자 수 |",
        "|---|---:|",
        f"| 원본 | {len(smiles):,} |",
        f"| 표준화 통과 (평가 데이터와 같은 설정) | {n_std:,} |",
        f"| InChIKey 중복 제거 | {n_dedup:,} |",
        f"| 평가 데이터와 골격 일치 제거 | {len(df):,} |",
        "",
        "표준화 거부 사유: " + (", ".join(f"{k} {v:,}" for k, v in rejected.most_common()) or "없음"),
        "",
        "평가 데이터셋별 골격(InChIKey 첫 블록) 일치 (한 분자가 여러 세트와 겹칠 수 있다): "
        + ", ".join(f"{k} {v}" for k, v in hit.items()) + f" → 합집합 {int(drop.sum())}개 제거",
        "",
        "평가 분할은 시드마다 바뀌어 'test 분자만' 뺄 수 없으므로, 평가 데이터 전체와 겹치는 골격을 뺐다.",
    ]
    card_path = ROOT / "docs" / "data_card_zinc250k_pretrain.md"
    card_path.write_text("\n".join(card) + "\n")
    print(json.dumps({k: v for k, v in ledger.items() if k != "standardize"}, ensure_ascii=False))
    print(f"→ {out.relative_to(ROOT)}, {card_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
