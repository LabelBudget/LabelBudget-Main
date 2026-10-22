"""SMILES 표준화. 스펙 02 §3 — 용해도 과제는 strip_salts=False.

순서 고정: 파싱 → Cleanup → [염 제거 | 다단편 거부] → Uncharger → 토토머 → 입체 → 필터.
모든 실패는 예외가 아니라 StandardizeResult.reason 으로 반환한다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

ORGANIC_ELEMENTS = frozenset({"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"})

REJECT_REASONS = (
    "empty_input", "parse_failed", "cleanup_failed", "multi_component",
    "fragment_choice_failed", "uncharge_failed",
    "non_organic_element", "no_carbon", "too_small", "too_large",
    "molwt_failed", "molwt_exceeded", "serialization_failed", "inchikey_failed",
)


class StereoPolicy(str, Enum):
    KEEP = "keep"
    STRIP = "strip"


@dataclass(frozen=True)
class StandardizeConfig:
    stereo: StereoPolicy = StereoPolicy.KEEP
    canonicalize_tautomer: bool = True
    strip_salts: bool = False              # 용해도는 반드시 False (03 §4)
    reject_multi_component: bool = True
    organic_only: bool = True
    min_heavy_atoms: int = 3
    max_heavy_atoms: int = 100
    max_mol_weight: float = 1000.0
    max_tautomers: int = 200
    max_transforms: int = 200

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stereo"] = self.stereo.value
        return d


@dataclass
class StandardizeResult:
    ok: bool
    smiles: str | None = None        # canonical
    inchikey: str | None = None
    reason: str | None = None        # 실패 사유 (REJECT_REASONS 중 하나)
    changed: bool = False            # 입력 대비 구조가 바뀌었는지
    notes: list[str] = field(default_factory=list)
    # 중복의 발생 단계를 가르기 위한 키 (pipeline 원장에서 사용)
    input_inchikey: str | None = None          # 표준화 전 입력 그대로
    pre_tautomer_inchikey: str | None = None   # 토토머 정규화만 뺀 결과


def _safe_inchikey(mol) -> str | None:
    try:
        return Chem.MolToInchiKey(mol) or None
    except Exception:
        return None


class Standardizer:
    """RDKit 표준화 객체를 한 번만 만들어 재사용한다."""

    def __init__(self, config: StandardizeConfig | None = None):
        self.config = config or StandardizeConfig()
        params = rdMolStandardize.CleanupParameters()
        params.maxTautomers = self.config.max_tautomers
        params.maxTransforms = self.config.max_transforms
        # RDKit 기본값은 토토머 정규화 중 입체를 지운다 (예: L-알라닌 → 라세미).
        # KEEP 정책에서 거울상이성질체가 인공 병합되지 않도록 끈다.
        if self.config.stereo is StereoPolicy.KEEP:
            params.tautomerRemoveSp3Stereo = False
            params.tautomerRemoveBondStereo = False
        self._params = params
        self._lfc = rdMolStandardize.LargestFragmentChooser()
        self._uncharger = rdMolStandardize.Uncharger(canonicalOrder=True)
        self._tautomer = rdMolStandardize.TautomerEnumerator(params)

    def __call__(self, smiles: str | None) -> StandardizeResult:
        try:
            return self._run(smiles)
        except Exception as e:  # 예외를 밖으로 던지지 않는다 (§3.6)
            return StandardizeResult(ok=False, reason="cleanup_failed", notes=[f"unexpected:{type(e).__name__}"])

    def _run(self, smiles: str | None) -> StandardizeResult:
        cfg = self.config
        notes: list[str] = []

        if smiles is None or not str(smiles).strip():
            return StandardizeResult(ok=False, reason="empty_input")
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return StandardizeResult(ok=False, reason="parse_failed")
        input_smiles = Chem.MolToSmiles(mol)
        input_key = _safe_inchikey(mol)

        try:
            mol = rdMolStandardize.Cleanup(mol, self._params)
        except Exception:
            return StandardizeResult(ok=False, reason="cleanup_failed")
        if mol is None:
            return StandardizeResult(ok=False, reason="cleanup_failed")
        if Chem.MolToSmiles(mol) != input_smiles:
            notes.append("cleanup_changed")

        if len(Chem.GetMolFrags(mol)) > 1:
            if cfg.strip_salts:
                try:
                    mol = self._lfc.choose(mol)
                except Exception:
                    return StandardizeResult(ok=False, reason="fragment_choice_failed")
                notes.append("salt_stripped")
            elif cfg.reject_multi_component:
                return StandardizeResult(ok=False, reason="multi_component")

        try:
            before = Chem.MolToSmiles(mol)
            mol = self._uncharger.uncharge(mol)
        except Exception:
            return StandardizeResult(ok=False, reason="uncharge_failed")
        if Chem.MolToSmiles(mol) != before:
            notes.append("uncharged")

        pre_tautomer = Chem.Mol(mol)
        if cfg.canonicalize_tautomer:
            try:
                before = Chem.MolToSmiles(mol)
                mol = self._tautomer.Canonicalize(mol)
                if Chem.MolToSmiles(mol) != before:
                    notes.append("tautomer_changed")
            except Exception:
                notes.append("tautomer_failed")  # 치명적이지 않다. 원본 유지

        if cfg.stereo is StereoPolicy.STRIP:
            before = Chem.MolToSmiles(mol)
            Chem.RemoveStereochemistry(mol)
            Chem.RemoveStereochemistry(pre_tautomer)
            if Chem.MolToSmiles(mol) != before:
                notes.append("stereo_stripped")

        if cfg.organic_only:
            symbols = {a.GetSymbol() for a in mol.GetAtoms()}
            if not symbols <= ORGANIC_ELEMENTS:
                return StandardizeResult(ok=False, reason="non_organic_element",
                                         notes=sorted(symbols - ORGANIC_ELEMENTS))
            if "C" not in symbols:
                return StandardizeResult(ok=False, reason="no_carbon")

        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < cfg.min_heavy_atoms:
            return StandardizeResult(ok=False, reason="too_small")
        if n_heavy > cfg.max_heavy_atoms:
            return StandardizeResult(ok=False, reason="too_large")

        try:
            molwt = Descriptors.MolWt(mol)
        except Exception:
            return StandardizeResult(ok=False, reason="molwt_failed")
        if molwt > cfg.max_mol_weight:
            return StandardizeResult(ok=False, reason="molwt_exceeded")

        try:
            out_smiles = Chem.MolToSmiles(mol)
        except Exception:
            out_smiles = None
        if not out_smiles:
            return StandardizeResult(ok=False, reason="serialization_failed")
        inchikey = _safe_inchikey(mol)
        if not inchikey:
            return StandardizeResult(ok=False, reason="inchikey_failed")

        pre_key = _safe_inchikey(pre_tautomer) if cfg.canonicalize_tautomer else inchikey
        return StandardizeResult(ok=True, smiles=out_smiles, inchikey=inchikey,
                                 changed=out_smiles != input_smiles, notes=notes,
                                 input_inchikey=input_key, pre_tautomer_inchikey=pre_key)


def standardize_smiles(smiles: str | None, config: StandardizeConfig | None = None) -> StandardizeResult:
    """단건 편의 함수. 대량 처리는 Standardizer 인스턴스를 재사용할 것."""
    return Standardizer(config)(smiles)
