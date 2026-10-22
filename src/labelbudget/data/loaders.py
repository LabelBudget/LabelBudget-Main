"""다운로드 캐시와 로딩. 파일이 없으면 DatasetUnavailable. 스펙 02 §1.2."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from labelbudget.data.sources import DatasetSource, get_source

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


class DatasetUnavailable(RuntimeError):
    """데이터 파일을 확보할 수 없을 때. 빈 DataFrame으로 실험이 돌아가는 것을 막는다."""


class DatasetShapeMismatch(ValueError):
    """원본 shape가 레지스트리의 실측값과 다를 때. 상류 파일 변경 가능성."""


def raw_path(name: str, raw_dir: Path | None = None) -> Path:
    return Path(raw_dir or DEFAULT_RAW_DIR) / get_source(name).filename


def _manual_message(src: DatasetSource, path: Path) -> str:
    lines = [f"'{src.name}'은 자동 다운로드를 지원하지 않습니다. 파일을 다음 위치에 배치하세요:", f"  {path}", "출처:"]
    lines += [f"  - {s}" for s in src.manual_sources]
    return "\n".join(lines)


def download(name: str, raw_dir: Path | None = None, *, force: bool = False, timeout: float = 60.0) -> Path:
    """캐시에 없으면 내려받는다. 수동 배치 데이터셋은 파일이 없으면 DatasetUnavailable."""
    src = get_source(name)
    path = raw_path(name, raw_dir)
    if path.exists() and not force:
        return path
    if src.is_manual:
        raise DatasetUnavailable(_manual_message(src, path))

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(src.url, timeout=timeout) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise DatasetUnavailable(f"'{src.name}' 다운로드 실패: {src.url}\n  {e}") from e
    tmp.replace(path)
    return path


def load_raw(
    name: str,
    raw_dir: Path | None = None,
    *,
    allow_download: bool = True,
    check_shape: bool = True,
) -> pd.DataFrame:
    """원본 CSV를 읽는다. shape를 실측값과 대조한 뒤 drop_cols를 제거한다."""
    src = get_source(name)
    path = download(name, raw_dir) if allow_download else raw_path(name, raw_dir)
    if not path.exists():
        raise DatasetUnavailable(_manual_message(src, path) if src.is_manual else f"'{src.name}' 파일 없음: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise DatasetUnavailable(f"'{src.name}' 파일이 비어 있습니다: {path}")
    if check_shape and df.shape != src.expected_shape:
        raise DatasetShapeMismatch(
            f"'{src.name}' shape {df.shape} != 실측 {src.expected_shape}. "
            "상류 파일이 바뀌었을 수 있습니다. 확인 후 check_shape=False로 진행하세요."
        )

    required = [src.smiles_col, *src.label_cols, *src.keep_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DatasetUnavailable(f"'{src.name}' 필수 컬럼 누락: {missing}")
    return df.drop(columns=[c for c in src.drop_cols if c in df.columns])


def load_labeled(
    name: str,
    label: str | None = None,
    raw_dir: Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """`smiles`, `label` + keep_cols 로 정리한다. 결측 라벨은 그대로 둔다 (원장에서 집계)."""
    src = get_source(name)
    label = label or src.default_label or (src.label_cols[0] if len(src.label_cols) == 1 else None)
    if label is None or label not in src.label_cols:
        raise ValueError(f"'{src.name}' 라벨을 지정하세요. 사용 가능: {list(src.label_cols)}")

    df = load_raw(name, raw_dir, **kwargs)
    out = pd.DataFrame({"smiles": df[src.smiles_col], "label": df[label]})
    for c in src.keep_cols:
        out[c] = df[c]
    out.attrs.update(dataset=src.name, task=src.task, label_name=label, label_unit=src.label_unit)
    return out
