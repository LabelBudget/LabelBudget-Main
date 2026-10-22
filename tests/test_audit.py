import pytest

from labelbudget.curate.audit import audit
from labelbudget.data.loaders import DEFAULT_RAW_DIR

pytestmark = pytest.mark.skipif(not (DEFAULT_RAW_DIR / "aqsoldb.csv").exists(), reason="AqSolDB 원본 없음")


@pytest.fixture(scope="module")
def report():
    return audit("aqsoldb", policies=["A_strip_no_taut", "B_reject_no_taut", "B_reject"])


def test_audit_reproduces_numbers(report):
    """스크립트 실측값 고정 (2026-10-22). 03 문서와 일치하는 항목은 주석으로 표시."""
    P = report["policies"]
    a, b0, b = P["A_strip_no_taut"], P["B_reject_no_taut"], P["B_reject"]

    assert report["raw_keys"]["file_inchikey_unique"] == 9982       # 03 §8 일치
    assert report["raw_keys"]["file_skeleton_unique"] == 9671       # 03 §8 일치
    assert report["extra"]["noise_floor"]["n_sd_positive"] == 2236  # 03 §6 일치
    assert report["extra"]["noise_floor"]["mean"] == pytest.approx(0.301, abs=5e-4)

    # 정책 A: 염 제거가 대규모 인공 중복을 만든다 (03: 191그룹, max 9.48)
    assert (a.n_pass, a.n_unique, a.n_created_groups) == (9554, 9225, 186)
    assert a.created_spread["max"] == pytest.approx(9.481, abs=1e-3)

    # 정책 B, 토토머 없음: 03 §5와 중복 그룹·spread 일치 (통과 건수는 03의 8,746과 다름)
    assert (b0.n_pass, b0.n_created_groups, b0.created_spread["n_gt_1.0"]) == (8735, 6, 1)
    assert b0.created_spread["median"] == pytest.approx(0.062, abs=1e-3)

    # 정책 B, 스펙 설정(토토머 포함)
    assert (b.n_pass, b.n_unique, b.n_created_groups, b.created_spread["n_gt_1.0"]) == (8735, 8709, 26, 4)


def test_policy_b_far_fewer_artificial_duplicates(report):
    P = report["policies"]
    assert P["B_reject_no_taut"].n_created_groups * 10 < P["A_strip_no_taut"].n_created_groups
