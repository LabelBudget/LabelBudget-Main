"""데이터셋 레지스트리 (URL, 컬럼 매핑). 스펙 02 §1."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TOX21_TASKS: tuple[str, ...] = (
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
)


@dataclass(frozen=True)
class DatasetSource:
    name: str
    task: Literal["regression", "classification"]
    filename: str
    smiles_col: str
    label_cols: tuple[str, ...]
    label_unit: str
    expected_shape: tuple[int, int]
    url: str | None = None                       # None이면 수동 배치
    default_label: str | None = None
    drop_cols: tuple[str, ...] = ()
    keep_cols: tuple[str, ...] = ()
    name_col: str | None = None                  # 감사 보고서용 화합물명
    manual_sources: tuple[str, ...] = field(default=())

    @property
    def is_manual(self) -> bool:
        return self.url is None


SOURCES: dict[str, DatasetSource] = {
    "aqsoldb": DatasetSource(
        name="aqsoldb",
        task="regression",
        filename="aqsoldb.csv",
        smiles_col="SMILES",
        label_cols=("Solubility",),
        label_unit="log10(mol/L)",
        expected_shape=(9982, 26),
        keep_cols=("SD", "Ocurrences", "Group"),
        name_col="Name",
        manual_sources=(
            "https://www.openml.org/d/43546  (브라우저로 다운로드)",
            "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8",
            "https://huggingface.co/datasets/maomlab/AqSolDB  (train/test를 합쳐서 쓸 것)",
        ),
    ),
    "esol": DatasetSource(
        name="esol",
        task="regression",
        filename="esol.csv",
        smiles_col="smiles",
        # 'ESOL predicted ...' 컬럼은 선행 모델의 예측값이므로 라벨로 쓰지 않는다
        label_cols=("measured log solubility in mols per litre",),
        label_unit="log10(mol/L)",
        expected_shape=(1128, 10),
        url="https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
        name_col="Compound ID",
    ),
    "tox21": DatasetSource(
        name="tox21",
        task="classification",
        filename="tox21.csv",
        smiles_col="smiles",
        label_cols=TOX21_TASKS,
        label_unit="binary",
        expected_shape=(7831, 15),
        url="https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest/tox21.csv",
        default_label="SR-ARE",
        drop_cols=("Unnamed: 0", "mol_id"),
    ),
    "lipophilicity": DatasetSource(
        name="lipophilicity",
        task="regression",
        filename="lipophilicity.csv",
        smiles_col="smiles",
        label_cols=("exp",),
        label_unit="log units",
        expected_shape=(4200, 4),
        url="https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest/Lipophilicity.csv",
        drop_cols=("Unnamed: 0",),
    ),
}


def get_source(name: str) -> DatasetSource:
    try:
        return SOURCES[name]
    except KeyError:
        raise KeyError(f"알 수 없는 데이터셋 '{name}'. 사용 가능: {sorted(SOURCES)}") from None
