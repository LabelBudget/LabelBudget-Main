import random

import pytest
from rdkit import Chem

from labelbudget.curate.standardize import (
    REJECT_REASONS,
    StandardizeConfig,
    Standardizer,
    StereoPolicy,
)


@pytest.fixture(scope="module")
def std():
    return Standardizer()


@pytest.fixture(scope="module")
def std_strip():
    return Standardizer(StandardizeConfig(strip_salts=True))


def _canon(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_default_config_does_not_strip_salts():
    assert StandardizeConfig().strip_salts is False


@pytest.mark.parametrize("smiles", ["CC(=O)[O-].[Na+]", "O=C([O-])c1ccccc1.[Na+]", "CCO.CCCCO"])
def test_reject_multi_component_by_default(std, smiles):
    r = std(smiles)
    assert not r.ok and r.reason == "multi_component"


def test_strip_salts_when_enabled(std_strip):
    r = std_strip("O=C([O-])c1ccccc1.[Na+]")
    assert r.ok and r.smiles == _canon("O=C(O)c1ccccc1")
    assert "salt_stripped" in r.notes


def test_standardize_charge(std):
    r = std("CC(=O)[O-]")
    assert r.ok and r.smiles == _canon("CC(=O)O")
    assert "uncharged" in r.notes


def test_standardize_tautomer(std):
    a, b = std("Oc1ccccn1"), std("O=c1cccc[nH]1")
    assert a.ok and b.ok
    assert a.smiles == b.smiles and a.inchikey == b.inchikey


@pytest.mark.parametrize("smiles", ["C[Sn](C)(C)Cl", "CC[Pb](CC)(CC)CC", "C[Hg]C"])
def test_standardize_reject_metal(std, smiles):
    r = std(smiles)
    assert not r.ok and r.reason == "non_organic_element"


def test_standardize_no_exception(std):
    rng = random.Random(0)
    alphabet = "CNOSPFIBrcl()[]=#@+-.1234567890%/\\*$!"
    broken = ["", "   ", None, "C1CC", "C(C", "c1cccc1", "[Xx]", "C(C)(C)(C)(C)C", "((((", "C==C"]
    broken += ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 30))) for _ in range(90)]
    assert len(broken) == 100
    for s in broken:
        r = std(s)
        assert r.ok or r.reason in REJECT_REASONS


def test_keep_stereo_survives_tautomer_canonicalization(std):
    l_ala, d_ala = std("C[C@H](N)C(=O)O"), std("C[C@@H](N)C(=O)O")
    assert "@" in l_ala.smiles
    assert l_ala.inchikey != d_ala.inchikey


def test_stereo_strip_merges_enantiomers():
    s = Standardizer(StandardizeConfig(stereo=StereoPolicy.STRIP))
    assert s("C[C@H](N)C(=O)O").inchikey == s("C[C@@H](N)C(=O)O").inchikey


@pytest.mark.parametrize(
    "smiles,reason",
    [("CO", "too_small"), ("OS(=O)(=O)O", "no_carbon"), ("C" * 101, "too_large")],
)
def test_filters(std, smiles, reason):
    assert std(smiles).reason == reason
