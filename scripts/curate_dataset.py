"""큐레이션 CLI. 스펙 02 §14-7.

python scripts/curate_dataset.py --dataset aqsoldb
python scripts/curate_dataset.py --dataset tox21 --task SR-ARE
"""
import argparse
import json
from pathlib import Path

import yaml

from labelbudget.curate.dedupe import ConflictRule
from labelbudget.curate.pipeline import curate, render_data_card
from labelbudget.curate.standardize import StandardizeConfig, StereoPolicy
from labelbudget.data.loaders import load_labeled
from labelbudget.data.sources import SOURCES, get_source

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    p.add_argument("--task", help="Tox21 과제 (기본: 레지스트리 default_label)")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    args = p.parse_args()

    conf = yaml.safe_load(args.config.read_text())
    ds_conf = conf["datasets"].get(args.dataset, {})
    std_conf = {**conf["standardize"], **ds_conf.get("standardize", {})}
    std_conf["stereo"] = StereoPolicy(std_conf["stereo"])
    src = get_source(args.dataset)

    df = load_labeled(args.dataset, label=args.task or ds_conf.get("target_task"))
    out, ledger = curate(
        df,
        dataset=args.dataset,
        task=src.task,
        label_name=df.attrs["label_name"],
        label_unit=df.attrs["label_unit"],
        standardize_config=StandardizeConfig(**std_conf),
        conflict_rule=ConflictRule(ds_conf.get("conflict_rule", "drop")),
        conflict_tolerance=ds_conf.get("conflict_tolerance", 0.5),
        keep_cols=src.keep_cols,
    )

    stem = args.dataset if src.task == "regression" else f"{args.dataset}_{df.attrs['label_name']}"
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    out.to_csv(processed / f"{stem}.csv", index=False)
    (processed / f"{stem}_ledger.json").write_text(json.dumps(ledger.to_dict(), indent=2, ensure_ascii=False))
    card = ROOT / "docs" / f"data_card_{stem}.md"
    card.write_text(render_data_card(ledger))

    print(f"{stem}: 원본 {ledger.n_raw:,} → 표준화 {ledger.n_after_standardize:,} → 최종 {ledger.n_after_dedupe:,}")
    print(f"  중복: 들어온 {ledger.n_duplicates_incoming} / 표준화가 만든 {ledger.n_duplicates_created_by_standardization}"
          f" (토토머 관여 {ledger.n_duplicates_created_by_tautomer})")
    print(f"  → {card.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
