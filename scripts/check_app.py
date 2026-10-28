"""데모 앱 점검: 모든 페이지와 예시 분자를 Streamlit AppTest로 실행한다. 배포 전에 돌린다.

~/miniforge3/envs/labelbudget-app/bin/python scripts/check_app.py      (앱 전용 환경)

- 페이지마다 예외가 없어야 한다
- 예시 분자마다: 거부 사례는 오류 메시지, 나머지는 예측 2개. 적용 범위 밖 사례는 경고
- 예시 분자의 예측과 실측값을 표로 출력한다 (README·안내용)
"""
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = str(ROOT / "app" / "streamlit_app.py")
PAGES = ["views/curves.py", "views/analysis.py", "views/predict.py"]


def fail(msg: str) -> None:
    print(f"실패: {msg}")
    sys.exit(1)


def main() -> int:
    at = AppTest.from_file(ENTRY, default_timeout=180).run()
    if at.exception:
        fail(f"개요: {at.exception}")
    for page in PAGES:
        at.switch_page(page).run()
        if at.exception:
            fail(f"{page}: {at.exception[0].message}")
        print(f"OK {page}")

    at = AppTest.from_file(ENTRY, default_timeout=180).run()
    at.switch_page("views/predict.py").run()
    ex = pd.read_csv(ROOT / "app" / "data" / "examples.csv")
    rows = []
    for e in ex.itertuples():
        at.selectbox[0].set_value(e.name).run()
        if at.exception:
            fail(f"{e.name}: {at.exception[0].message}")
        if e.note.startswith("거부"):
            if not at.error:
                fail(f"{e.name}: 거부되지 않음")
            rows.append({"이름": e.name, "결과": at.error[0].value.split(".")[0]})
            continue
        if len(at.metric) != 2:
            fail(f"{e.name}: 예측 {len(at.metric)}개")
        if "적용 범위" in e.note and not at.warning:
            fail(f"{e.name}: 적용 범위 경고 없음")
        b, d = (float(m.value.split()[0]) for m in at.metric)
        rows.append({"이름": e.name, "SMILES": e.smiles, "실측": e.measured, "구분": e.role, "B": b, "D": d,
                     "경고": "적용 범위 밖" if at.warning else ""})
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
