import pandas as pd
import pytest

from labelbudget.curate.dedupe import ConflictRule, deduplicate, leakage_check
from labelbudget.curate.standardize import StandardizeConfig, Standardizer

# 같은 골격(첫 블록), 다른 입체 블록
K_A = "QNAYBMKLOCPYGJ-REOHCLBHSA-N"
K_A2 = "QNAYBMKLOCPYGJ-UWTATZPHSA-N"
K_B = "WPYMKLBDIGXBTP-UHFFFAOYSA-N"


def _df(keys, labels):
    return pd.DataFrame({"inchikey": keys, "label": labels, "smiles": [f"s{i}" for i in range(len(keys))]})


def test_dedupe_exact():
    out, rep = deduplicate(_df([K_B, K_B, K_B], [1.0, 1.1, 1.2]))
    assert len(out) == 1 and out.label.iloc[0] == pytest.approx(1.1)
    assert out.n_merged.iloc[0] == 3
    assert rep.n_exact_duplicate_rows == 2 and rep.n_conflicting_molecules == 0


def test_dedupe_conflict_drop():
    out, rep = deduplicate(_df([K_B, K_B, K_A], [0.0, 2.0, 5.0]), rule=ConflictRule.DROP, tolerance=0.5)
    assert out.inchikey.tolist() == [K_A]
    assert rep.n_conflicting_molecules == 1 and rep.n_dropped_by_conflict == 1
    assert rep.examples[0]["spread"] == pytest.approx(2.0) and rep.examples[0]["dropped"]


def test_dedupe_conflict_within_tolerance_is_merged():
    out, rep = deduplicate(_df([K_B, K_B], [0.0, 0.4]), tolerance=0.5)
    assert len(out) == 1 and rep.n_dropped_by_conflict == 0


def test_dedupe_conflict_median_keeps():
    out, _ = deduplicate(_df([K_B, K_B, K_B], [0.0, 1.0, 5.0]), rule=ConflictRule.MEDIAN)
    assert out.label.iloc[0] == pytest.approx(1.0)


def test_dedupe_majority_and_tie():
    df = _df([K_B, K_B, K_B, K_A, K_A], [1, 1, 0, 1, 0])
    out, rep = deduplicate(df, task="classification", rule=ConflictRule.MAJORITY)
    assert out.set_index("inchikey").label.to_dict() == {K_B: 1}
    assert rep.n_dropped_by_conflict == 1


def test_dedupe_rule_task_mismatch():
    with pytest.raises(ValueError):
        deduplicate(_df([K_B], [1]), task="regression", rule=ConflictRule.MAJORITY)


def test_dedupe_skeleton_report():
    out, rep = deduplicate(_df([K_A, K_A2, K_B], [0.0, 3.0, 1.0]))
    assert len(out) == 3  # 제거하지 않는다
    assert rep.n_skeleton_collision_groups == 1 and rep.n_skeleton_collision_molecules == 2


def test_salt_merge_regression():
    """정책 근거의 회귀 테스트: 염 제거를 켜면 탄산나트륨과 탄산칼슘이 같은 키가 된다 (03 §4)."""
    na2co3, caco3 = "O=C([O-])[O-].[Na+].[Na+]", "O=C([O-])[O-].[Ca+2]"
    strip = Standardizer(StandardizeConfig(strip_salts=True, organic_only=False))
    assert strip(na2co3).inchikey == strip(caco3).inchikey
    keep = Standardizer(StandardizeConfig())
    assert keep(na2co3).reason == keep(caco3).reason == "multi_component"


def test_leakage_check():
    r = leakage_check([K_A, K_B], [K_A2])
    assert r == {"exact_overlap": 0, "skeleton_overlap": 1, "clean": True}
    assert not leakage_check([K_A], [K_A, K_B])["clean"]
