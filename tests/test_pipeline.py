import pandas as pd
import pytest

from labelbudget.curate.dedupe import ConflictRule
from labelbudget.curate.pipeline import curate, render_data_card
from labelbudget.data.loaders import DEFAULT_RAW_DIR, load_labeled
from labelbudget.data.sources import get_source


def _run(smiles, labels, **kw):
    df = pd.DataFrame({"smiles": smiles, "label": labels})
    return curate(df, dataset="toy", task=kw.pop("task", "regression"), label_name="y", label_unit="u", **kw)


def test_ledger_counts_and_missing():
    out, L = _run(["CCO", None, "CC(=O)[O-].[Na+]", "CCCC"], [0.1, 0.2, 0.3, None])
    assert (L.n_raw, L.n_after_missing, L.n_after_standardize, L.n_after_dedupe) == (4, 2, 1, 1)
    assert L.reject_reasons == {"multi_component": 1}


def test_duplicate_origins_split():
    smiles = [
        "CCCO", "OCCC",                 # 들어온 중복 (같은 분자의 다른 표기 → 입력 키 동일)
        "CC(=O)[O-]", "CC(=O)O",        # 표준화(중성화)가 만든 중복
        "CC(C)=O", "C=C(C)O",           # 토토머가 만든 중복 (케토-엔올은 표준 InChI가 구분한다)
    ]
    _, L = _run(smiles, [0.0, 0.1, 1.0, 1.1, 2.0, 2.1])
    assert L.n_duplicates_incoming == 1
    assert L.n_duplicates_created_by_standardization == 2
    assert L.n_duplicates_created_by_tautomer == 1
    assert L.duplicate_label_spread_stats["tautomer"]["n_groups"] == 1
    assert len(L.tautomer_duplicate_examples) == 1


def test_tautomer_off_creates_no_tautomer_duplicates():
    from labelbudget.curate.standardize import StandardizeConfig
    _, L = _run(["CC(C)=O", "C=C(C)O"], [2.0, 2.1],
                standardize_config=StandardizeConfig(canonicalize_tautomer=False))
    assert L.n_duplicates_created_by_tautomer == 0 and L.n_after_dedupe == 2


def test_inchi_mobile_h_tautomers_count_as_incoming():
    """표준 InChI는 이동 수소 토토머(2-하이드록시피리딘/2-피리돈)를 이미 같은 키로 본다."""
    _, L = _run(["Oc1ccccn1", "O=c1cccc[nH]1"], [2.0, 2.1])
    assert L.n_duplicates_incoming == 1 and L.n_duplicates_created_by_tautomer == 0


def test_imbalance_warning_in_card():
    smiles = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO", "CCCCCCCO", "c1ccccc1"]
    _, L = _run(smiles, [0, 0, 0, 0, 0, 0, 1], task="classification", conflict_rule=ConflictRule.MAJORITY)
    card = render_data_card(L)
    assert "불균형" in card and "AUPRC" in card


def test_card_sections_present():
    _, L = _run(["CC(C)=O", "C=C(C)O", "CCO"], [2.0, 2.1, 0.0])
    card = render_data_card(L)
    for h in ["## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "토토머 정규화가 관여"]:
        assert h in card


def _e2e(name, label=None):
    if not (DEFAULT_RAW_DIR / get_source(name).filename).exists():
        pytest.skip("원본 파일 없음")
    src = get_source(name)
    df = load_labeled(name, label=label, allow_download=False)
    rule = ConflictRule.MAJORITY if src.task == "classification" else ConflictRule.DROP
    out, L = curate(df, dataset=name, task=src.task, label_name=df.attrs["label_name"],
                    label_unit=df.attrs["label_unit"], conflict_rule=rule)
    assert 0 < L.n_after_dedupe <= L.n_raw and out.inchikey.is_unique
    assert render_data_card(L).startswith(f"# 데이터 카드 — {name}")
    return L


def test_e2e_esol():
    _e2e("esol")


def test_e2e_tox21_sr_are():
    L = _e2e("tox21", "SR-ARE")
    assert L.label_name == "SR-ARE"
