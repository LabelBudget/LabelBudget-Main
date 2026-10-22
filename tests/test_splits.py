import numpy as np
import pandas as pd
import pytest

from labelbudget.curate.dedupe import leakage_check
from labelbudget.evaluation.budget import LowPositiveWarning, subsample_train
from labelbudget.splits.strategies import iter_splits, make_split, scaffold_groups

RINGS = ["c1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCNCC1", "c1ccoc1", "c1ccsc1", "C1CCOC1",
         "c1ccc2ccccc2c1", "C1CC1", "C1CCC1", "c1cnccn1", "c1ccc2[nH]ccc2c1"]


@pytest.fixture(scope="module")
def df():
    # 고리 2개 × 연결 사슬 조합 + 비고리 분자 → 실제 데이터처럼 작은 골격 그룹이 많은 300개
    rng = np.random.default_rng(0)
    smiles = []
    for i in range(300):
        if i % 10 == 0:
            smiles.append("C" * (1 + i % 7) + "O")
            continue
        r1 = RINGS[rng.integers(len(RINGS))]
        r2 = RINGS[rng.integers(len(RINGS))].replace("1", "3").replace("2", "4")
        smiles.append(f"C{'C' * (i % 3)}{r1}{'C' * rng.integers(4)}{r2}")
    keys = [f"KEY{i:05d}-X-N" for i in range(len(smiles))]
    return pd.DataFrame({"smiles": smiles, "inchikey": keys, "label": rng.normal(size=len(smiles))})


def _sets(s):
    return set(s.train), set(s.valid), set(s.test)


def test_split_partitions_everything(df):
    for s in iter_splits(df, strategies=["random", "scaffold", "scaffold_random"], seeds=[0, 1]):
        tr, va, te = _sets(s)
        assert not (tr & va or tr & te or va & te)
        assert tr | va | te == set(range(len(df)))


def test_split_scaffold_deterministic(df):
    a, b = make_split(df, strategy="scaffold", seed=0), make_split(df, strategy="scaffold", seed=99)
    assert all(np.array_equal(x, y) for x, y in zip((a.train, a.valid, a.test), (b.train, b.valid, b.test)))


def test_split_scaffold_random_varies(df):
    tests = {tuple(make_split(df, strategy="scaffold_random", seed=s).test) for s in range(5)}
    assert len(tests) > 1


def test_scaffold_groups_not_split_across_sets(df):
    groups, _ = scaffold_groups(df.smiles)
    for strategy in ("scaffold", "scaffold_random"):
        s = make_split(df, strategy=strategy, seed=3)
        where = {i: k for k, idx in zip("tvs", (s.train, s.valid, s.test)) for i in idx}
        for members in groups.values():
            assert len({where[i] for i in members}) == 1


def test_iter_splits_collapses_deterministic(df):
    got = [s.strategy for s in iter_splits(df, strategies=["random", "scaffold", "scaffold_random"], seeds=range(4))]
    assert got.count("scaffold") == 1 and got.count("random") == 4 and got.count("scaffold_random") == 4


def test_split_no_leakage(df):
    for s in iter_splits(df, strategies=["random", "scaffold", "scaffold_random"], seeds=[0, 1, 2]):
        assert leakage_check(df.inchikey.iloc[s.train], df.inchikey.iloc[s.test])["exact_overlap"] == 0


def test_split_fractions_reasonable(df):
    for strategy in ("random", "scaffold_random"):
        s = make_split(df, strategy=strategy, seed=0)
        assert abs(len(s.train) / len(df) - 0.8) < 0.1
        assert len(s.test) > 0 and len(s.valid) > 0


def test_split_time_boundary():
    years = [2000] * 5 + [2001] * 10 + [2002] * 3 + [2003] * 2
    d = pd.DataFrame({"smiles": ["C"] * 20, "year": years})
    s = make_split(d, strategy="time", seed=0, date_col="year", frac_train=0.5, frac_valid=0.25, frac_test=0.25)
    yr = d.year.to_numpy()
    assert set(yr[s.train]).isdisjoint(yr[s.valid]) and set(yr[s.train]).isdisjoint(yr[s.test])
    assert set(yr[s.valid]).isdisjoint(yr[s.test])
    assert yr[s.train].max() < yr[s.test].min()


def test_split_time_requires_date(df):
    with pytest.raises(ValueError, match="date_col"):
        make_split(df, strategy="time", seed=0)


def test_split_time_rejects_missing_dates():
    d = pd.DataFrame({"smiles": ["C"] * 4, "year": [2000, None, 2001, 2002]})
    with pytest.raises(ValueError, match="결측"):
        make_split(d, strategy="time", seed=0, date_col="year")


def test_budget_stratify():
    y = np.array([1] * 160 + [0] * 840)
    idx = subsample_train(y, n_labels=300, seed=0, task="classification")
    assert len(idx) == 300 and len(set(idx)) == 300
    assert y[idx].mean() == pytest.approx(0.16, abs=0.005)


def test_budget_full_when_larger():
    y = np.zeros(50)
    assert np.array_equal(subsample_train(y, n_labels=100, seed=0, task="regression"), np.arange(50))
    assert np.array_equal(subsample_train(y, n_labels=None, seed=0, task="regression"), np.arange(50))


def test_budget_seed_independence(df):
    s = make_split(df, strategy="scaffold_random", seed=7)
    y_tr = df.label.to_numpy()[s.train]
    subs = [s.train[subsample_train(y_tr, n_labels=50, seed=k, task="regression")] for k in range(3)]
    assert len({tuple(x) for x in subs}) == 3                      # 서브샘플은 달라지고
    assert all(set(x) <= set(s.train) for x in subs)                # 학습셋 밖으로 나가지 않으며
    assert np.array_equal(make_split(df, strategy="scaffold_random", seed=7).test, s.test)  # 테스트셋 불변


def test_budget_low_positive_warning():
    y = np.array([1] * 50 + [0] * 950)
    with pytest.warns(LowPositiveWarning):
        subsample_train(y, n_labels=300, seed=0, task="classification")
