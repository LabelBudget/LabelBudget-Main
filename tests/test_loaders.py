import pandas as pd
import pytest

from labelbudget.data.loaders import (
    DEFAULT_RAW_DIR,
    DatasetShapeMismatch,
    DatasetUnavailable,
    download,
    load_labeled,
    load_raw,
)
from labelbudget.data.sources import SOURCES, TOX21_TASKS, get_source


def _write_fake_aqsoldb(raw_dir, n_rows=3):
    cols = ["ID", "Name", "SMILES", "Solubility", "SD", "Ocurrences", "Group"]
    df = pd.DataFrame({c: range(n_rows) for c in cols})
    df.to_csv(raw_dir / "aqsoldb.csv", index=False)


def test_registry_tox21_default_is_sr_are():
    src = get_source("tox21")
    assert src.default_label == "SR-ARE"
    assert set(src.label_cols) == set(TOX21_TASKS) and len(TOX21_TASKS) == 12


def test_registry_esol_uses_measured_not_predicted():
    (label,) = get_source("esol").label_cols
    assert label.startswith("measured") and "predicted" not in label


def test_registry_unknown_name():
    with pytest.raises(KeyError):
        get_source("nope")


def test_manual_dataset_missing_raises(tmp_path):
    with pytest.raises(DatasetUnavailable, match="aqsoldb.csv"):
        download("aqsoldb", tmp_path)
    with pytest.raises(DatasetUnavailable):
        load_raw("aqsoldb", tmp_path)


def test_no_download_when_disallowed(tmp_path):
    with pytest.raises(DatasetUnavailable):
        load_raw("esol", tmp_path, allow_download=False)


def test_shape_mismatch_raises(tmp_path):
    _write_fake_aqsoldb(tmp_path)
    with pytest.raises(DatasetShapeMismatch):
        load_raw("aqsoldb", tmp_path)


def test_load_labeled_maps_columns(tmp_path):
    _write_fake_aqsoldb(tmp_path)
    df = load_labeled("aqsoldb", raw_dir=tmp_path, check_shape=False)
    assert list(df.columns) == ["smiles", "label", "SD", "Ocurrences", "Group"]
    assert df.attrs["task"] == "regression"


def test_missing_required_column_raises(tmp_path):
    pd.DataFrame({"SMILES": ["C"], "Solubility": [0.0]}).to_csv(tmp_path / "aqsoldb.csv", index=False)
    with pytest.raises(DatasetUnavailable, match="필수 컬럼"):
        load_raw("aqsoldb", tmp_path, check_shape=False)


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_real_files_match_registry(name):
    if not (DEFAULT_RAW_DIR / get_source(name).filename).exists():
        pytest.skip("원본 파일 없음")
    df = load_raw(name, allow_download=False)
    src = get_source(name)
    assert df.shape[0] == src.expected_shape[0]
    assert not any(c in df.columns for c in src.drop_cols)
