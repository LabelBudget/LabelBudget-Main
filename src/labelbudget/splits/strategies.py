"""random / scaffold / scaffold_random / time 분할. 스펙 02 §6.

scaffold는 결정론적이라 시드를 바꿔도 결과가 같다. 시드 분산 측정은 scaffold_random으로 한다.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

SplitName = Literal["random", "scaffold", "scaffold_random", "time"]
DETERMINISTIC: frozenset[str] = frozenset({"scaffold", "time"})


@dataclass(frozen=True)
class Split:
    train: np.ndarray   # 위치 인덱스 (0-based)
    valid: np.ndarray
    test: np.ndarray
    strategy: str
    seed: int
    meta: dict = field(default_factory=dict)


def _targets(n: int, frac_train: float, frac_valid: float, frac_test: float) -> tuple[int, int]:
    if not np.isclose(frac_train + frac_valid + frac_test, 1.0):
        raise ValueError("frac_train + frac_valid + frac_test 는 1이어야 합니다")
    n_train = int(round(frac_train * n))
    n_valid = int(round(frac_valid * n))
    return n_train, n_valid


def scaffold_groups(smiles: Iterable[str]) -> tuple[dict[str, list[int]], int]:
    """Bemis-Murcko 골격별 위치 인덱스. 골격 추출 실패분은 각자 단독 그룹."""
    groups: dict[str, list[int]] = defaultdict(list)
    n_failed = 0
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        try:
            key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else None
        except Exception:
            key = None
        if key is None:
            n_failed += 1
            key = f"__unparsed_{i}__"
        groups[key].append(i)
    return dict(groups), n_failed


def _fill(ordered: list[list[int]], n_train: int, n_valid: int) -> tuple[list[int], list[int], list[int]]:
    train, valid, test = [], [], []
    for g in ordered:
        if len(train) + len(g) <= n_train:
            train += g
        elif len(valid) + len(g) <= n_valid:
            valid += g
        else:
            test += g
    return train, valid, test


def _scaffold_split(smiles, seed, n_train, n_valid, n_test, randomize):
    groups, n_failed = scaffold_groups(smiles)
    items = list(groups.items())
    if randomize:
        # Chemprop balanced scaffold: test 목표의 절반보다 큰 그룹은 먼저 train으로, 나머지만 섞는다
        big = sorted([kv for kv in items if len(kv[1]) > n_test / 2], key=lambda kv: (-len(kv[1]), kv[0]))
        small = sorted([kv for kv in items if len(kv[1]) <= n_test / 2], key=lambda kv: kv[0])
        rng = np.random.default_rng(seed)
        small = [small[i] for i in rng.permutation(len(small))]
        ordered = big + small
    else:
        ordered = sorted(items, key=lambda kv: (-len(kv[1]), kv[0]))
    train, valid, test = _fill([g for _, g in ordered], n_train, n_valid)
    largest = max(items, key=lambda kv: len(kv[1]))
    meta = {"n_scaffold_groups": len(groups), "n_scaffold_parse_failed": n_failed,
            "largest_group": {"scaffold": largest[0], "size": len(largest[1])}}
    return train, valid, test, meta


def _time_split(dates: pd.Series, n_train: int, n_valid: int):
    if dates.isna().any():
        raise ValueError(f"time 분할: 날짜 결측 {int(dates.isna().sum())}건. 결측을 허용하지 않습니다")
    order = np.argsort(dates.to_numpy(), kind="stable")
    sorted_dates = dates.to_numpy()[order]

    def push(cut: int) -> int:
        # 같은 날짜가 경계에 걸치면 그 값이 끝나는 곳까지 민다
        while 0 < cut < len(sorted_dates) and sorted_dates[cut] == sorted_dates[cut - 1]:
            cut += 1
        return cut

    c1 = push(n_train)
    c2 = push(max(c1, n_train + n_valid))
    train, valid, test = order[:c1].tolist(), order[c1:c2].tolist(), order[c2:].tolist()

    def span(idx):
        return [str(dates.iloc[idx].min()), str(dates.iloc[idx].max())] if idx else None

    meta = {"train_range": span(train), "valid_range": span(valid), "test_range": span(test)}
    return train, valid, test, meta


def make_split(
    df: pd.DataFrame,
    *,
    strategy: SplitName,
    seed: int,
    smiles_col: str = "smiles",
    date_col: str | None = None,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
) -> Split:
    n = len(df)
    n_train, n_valid = _targets(n, frac_train, frac_valid, frac_test)
    n_test = n - n_train - n_valid
    meta: dict = {}

    if strategy == "random":
        perm = np.random.default_rng(seed).permutation(n)
        train, valid, test = perm[:n_train], perm[n_train:n_train + n_valid], perm[n_train + n_valid:]
    elif strategy in ("scaffold", "scaffold_random"):
        train, valid, test, meta = _scaffold_split(df[smiles_col].tolist(), seed, n_train, n_valid, n_test,
                                                   randomize=strategy == "scaffold_random")
    elif strategy == "time":
        if date_col is None or date_col not in df:
            raise ValueError("time 분할에는 date_col이 필요합니다. ESOL/Tox21/AqSolDB/Lipophilicity에는 날짜 컬럼이 없습니다 "
                             "(ChEMBL 추출분에만 적용)")
        train, valid, test, meta = _time_split(df[date_col].reset_index(drop=True), n_train, n_valid)
    else:
        raise ValueError(f"알 수 없는 분할 '{strategy}'")

    as_arr = lambda x: np.sort(np.asarray(x, dtype=np.int64))  # noqa: E731
    return Split(as_arr(train), as_arr(valid), as_arr(test), strategy, seed, meta)


def iter_splits(df: pd.DataFrame, *, strategies: Iterable[str], seeds: Iterable[int], **kwargs) -> Iterator[Split]:
    """결정론적 분할(scaffold, time)은 시드 루프를 1회로 축약한다."""
    seeds = list(seeds)
    for strategy in strategies:
        for seed in (seeds[:1] if strategy in DETERMINISTIC else seeds):
            yield make_split(df, strategy=strategy, seed=seed, **kwargs)
