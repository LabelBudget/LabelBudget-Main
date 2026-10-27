"""analyze_d.py 전체 경로 점검. 실제 B·C v2 결과를 복사해 가짜 C_fixed·D 결과를 만든다 (실제 D 결과를 쓰지 않는다)."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"


def _load_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("analyze_d", ROOT / "scripts" / "analyze_d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not (RES / "aqsoldb_gnn_v2_full.csv").exists(), reason="C v2 결과 없음")
def test_analyze_d_end_to_end(tmp_path, monkeypatch):
    mod = _load_module()
    res = tmp_path / "reports" / "results"
    res.mkdir(parents=True)
    shutil.copy(RES / "aqsoldb_baselines_full.csv", res)
    v2 = pd.read_csv(RES / "aqsoldb_gnn_v2_full.csv", dtype={"n_labels": str})
    v2.to_csv(res / "aqsoldb_gnn_v2_full.csv", index=False)
    rng = np.random.default_rng(0)
    for m, tag, shift in [("C_fixed", "fixed", 0.05), ("D_mask", "dmask", -0.05), ("D_contrast", "dcontrast", 0.0)]:
        g = v2.copy()
        g["model"] = m
        g["metric_value"] = g.metric_value + shift + rng.normal(0, 0.01, len(g))
        g["hp_valid_score"] = g.hp_valid_score + shift
        g = g[g.hp_n_trials != np.where(g.n_labels.isin(["3000", "full"]), 20, -1)]  # 큰 예산은 10회까지만 있다
        g.to_csv(res / f"aqsoldb_gnn_{tag}_full.csv", index=False)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "RES", res)
    monkeypatch.setattr(sys, "argv", ["analyze_d.py", "--dataset", "aqsoldb"])
    assert mod.main() == 0
    md = (tmp_path / "reports" / "d_aqsoldb.md").read_text()
    for s in ("## 2. 사전 등록 비교", "**H1**", "**H2**", "**H4**", "C v2 − C_fixed", "## 5. 보조"):
        assert s in md
    pick = json.loads((res / "aqsoldb_best_d.json").read_text())
    assert pick["best_d"] == "D_mask"  # 회귀: valid RMSE가 0.05 작은 쪽
    assert (tmp_path / "reports" / "figures" / "aqsoldb_budget_curve_d.png").exists()

    # H3: random 분할은 C v2 + 고른 D 하나만 있다. best_d.json을 다시 쓰지 않고 두 분할의 H2를 나란히 적는다
    for m, tag in [("C_dmpnn_v2", "v2"), ("D_mask", "dmask")]:
        g = pd.read_csv(res / f"aqsoldb_gnn_{tag}_full.csv", dtype={"n_labels": str})
        g = g[g.hp_n_trials != np.where(g.n_labels.isin(["3000", "full"]), 20, -1)]
        g.assign(split_strategy="random").to_csv(res / f"aqsoldb_gnn_{tag}_random_full.csv", index=False)
    before = (res / "aqsoldb_best_d.json").read_text()
    monkeypatch.setattr(sys, "argv", ["analyze_d.py", "--dataset", "aqsoldb", "--split", "random"])
    assert mod.main() == 0
    md = (tmp_path / "reports" / "d_aqsoldb_random.md").read_text()
    assert "## 4. H3" in md and "| D  마스킹 사전학습 | " in md and "C_fixed" not in md.split("## 4. H3")[1]
    assert (res / "aqsoldb_best_d.json").read_text() == before


def test_trials_for_budget():
    mod = _load_module()
    assert [mod.trials_for(n) for n in ["100", "300", "1000", "3000", "full"]] == [20, 20, 20, 10, 10]
