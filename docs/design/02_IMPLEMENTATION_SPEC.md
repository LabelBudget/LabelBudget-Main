# LabelBudget — 구현 스펙

> 이 문서만 읽고 1~3주차를 구현할 수 있도록 작성했다.

- 최종 갱신: 2026-10-22
- 대상: **1주차(데이터 큐레이션) · 2주차(평가 프레임워크) · 3주차(베이스라인 A·B)**
- 선행 문서: `01_RESEARCH_PLAN.md` (연구 배경), **`03_AQSOLDB_AUDIT.md` (설계 변경 근거 — 필수)**
- 4주차 이후(MPNN 직접 구현, 사전학습)는 1~3주차 결과를 보고 별도 스펙으로 작성

> ⚠️ 이 문서의 §3.2(염 처리 정책)와 §4.2(충돌 처리)는 **실측 감사로 변경된 내용**이다.
> `03_AQSOLDB_AUDIT.md`를 읽지 않고 구현하면 이유를 모른 채 따라가게 된다.

---

## 0. 검증된 환경

실제로 설치·실행해 확인한 값이다. 추측이 아니다.

```
Python       3.13.16
rdkit        2026.09.1     (pip install rdkit)
lightgbm     4.7.0         (pip install lightgbm)
scikit-learn 1.9.1
scipy        1.18.1
pandas       3.0.5
numpy        2.5.3
```

`requirements.txt`:

```
rdkit>=2026.09
lightgbm>=4.7
scikit-learn>=1.9
scipy>=1.18
pandas>=2.2
numpy>=2.0
pyyaml>=6.0
pytest>=8.0
# 4주차 이후
# torch>=2.4
# torch-geometric>=2.6
```

### 실측 성능 (구현 계획 수립용)

| 작업 | 실측 |
|---|---|
| 표준화 (토토머 제외) 9,982건 | 9.5초 |
| 토토머 정규화 | 8 ms/분자 |
| 토토머 정규화 9,982건 | 약 1.4분 |

토토머 정규화는 느리지 않다. **성능을 이유로 생략하지 말 것.**

---

## 1. 데이터

### 1.1 자동 다운로드 (URL·행수·컬럼 전부 실측 검증)

| 데이터셋 | 실측 shape | URL |
|---|---|---|
| ESOL | **1,128 × 10** | `https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv` |
| Tox21 | **7,831 × 15** | `https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest/tox21.csv` |
| Lipophilicity | **4,200 × 4** | `https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest/Lipophilicity.csv` |

**차단 확인된 호스트** (시도하지 말 것)

```
deepchemdata.s3.us-west-1.amazonaws.com   # deepchem S3 미러 전체
dataverse.harvard.edu                      # Harvard Dataverse
www.openml.org / api.openml.org            # OpenML API
```

### 1.2 AqSolDB — 수동 배치 (확보 완료)

파일 `curated-solubility-dataset.csv` (**9,982 × 26**)을 확보해 실측 감사까지 마쳤다.

- 배치 경로: **`data/raw/aqsoldb.csv`** (출처별 원본 파일명이 달라 이 이름으로 통일)
- 재확보가 필요할 경우의 출처:

| 우선 | 출처 | URL |
|---|---|---|
| 1순위 | OpenML (브라우저로 다운로드) | `https://www.openml.org/d/43546` |
| 2순위 | Harvard Dataverse (논문 원본) | `https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8` |
| 3순위 | HuggingFace (train/test 분리 — **합쳐서 쓸 것**) | `https://huggingface.co/datasets/maomlab/AqSolDB` |

> 로더는 파일이 없을 때 **조용히 빈 DataFrame을 반환하지 말고** `DatasetUnavailable`
> 예외를 던지며 배치 위치를 안내해야 한다. 데이터가 없는 상태로 실험이 돌아가는
> 것이 가장 위험하다.

### 1.3 컬럼 매핑 (전부 실측 확인)

| 데이터셋 | SMILES | 라벨 | 단위 | 비고 |
|---|---|---|---|---|
| AqSolDB | `SMILES` | `Solubility` | log10(mol/L) | `SD`·`Ocurrences`·`Group` 보존 필수 |
| ESOL | `smiles` | `measured log solubility in mols per litre` | log10(mol/L) | `ESOL predicted...` 컬럼은 **예측값이므로 쓰지 말 것** |
| Tox21 | `smiles` | 12과제 중 선택 | binary | **`Unnamed: 0`, `mol_id` 컬럼 제거** |
| Lipophilicity | `smiles` | `exp` | log units | **`Unnamed: 0` 제거** |

> ESOL에는 `measured`(실측)와 `ESOL predicted`(선행 모델 예측) 두 컬럼이 있다.
> 후자를 라벨로 쓰면 다른 모델의 출력을 학습하는 것이 된다. 반드시 `measured`를 쓴다.

### 1.4 ⚠️ Tox21 과제 선택 — 분류 실험의 전제

Tox21 12과제의 **실측** 유효 건수와 양성률이다.

| 과제 | 유효 n | 양성 | 양성률 | 불균형 |
|---|---:|---:|---:|---:|
| **SR-ARE** | **5,832** | **942** | **16.15%** | **5.2:1** |
| SR-MMP | 5,810 | 918 | 15.80% | 5.3:1 |
| NR-ER | 6,193 | 793 | 12.80% | 6.8:1 |
| NR-AhR | 6,549 | 768 | 11.73% | 7.5:1 |
| SR-p53 | 6,774 | 423 | 6.24% | 15:1 |
| SR-HSE | 6,467 | 372 | 5.75% | 16:1 |
| NR-Aromatase | 5,821 | 300 | 5.15% | 18:1 |
| NR-ER-LBD | 6,955 | 350 | 5.03% | 19:1 |
| NR-AR | 7,265 | 309 | 4.25% | 22:1 |
| SR-ATAD5 | 7,072 | 264 | 3.73% | 26:1 |
| NR-AR-LBD | 6,758 | 237 | 3.51% | 27:1 |
| NR-PPAR-gamma | 6,450 | 186 | 2.88% | 34:1 |

**주 분류 과제는 `SR-ARE`로 고정한다.** 12과제 중 가장 균형적이고 유효 건수도 충분하다.

라벨 예산별 양성 개수 (SR-ARE, train 80% 기준):

| 라벨 예산 | 양성 개수 | 판정 |
|---:|---:|---|
| 100 | 약 16 | ❌ **사용 금지** |
| 300 | 약 48 | ⚠️ 하한 |
| 1,000 | 약 162 | ✅ |
| 3,000 | 약 485 | ✅ |
| 전체(약 4,665) | 약 753 | ✅ |

> 양성 20개 미만에서 AUPRC는 사실상 무작위 수준으로 요동친다.
> **분류 예산 사다리는 `[300, 1000, 3000, None]`으로 한다.** 회귀는 100부터 쓴다.
> `NR-AR`을 기본값으로 두면 예산 100에서 양성이 4개뿐이라 실험이 성립하지 않는다.

Tox21 12과제 목록:
`NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53`

단일 과제 실험에서는 해당 과제가 결측인 행을 제거한다. 멀티태스크로 갈 경우 마스크 손실이 필요하다(4주차 이후).

---

## 2. 디렉터리 구조

```
labelbudget/
├── README.md
├── requirements.txt
├── reproduce.sh
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/                  # 다운로드 캐시 (.gitignore)
│   └── processed/            # 큐레이션 결과 (.gitignore)
├── src/labelbudget/
│   ├── curate/
│   │   ├── standardize.py    # §3
│   │   ├── dedupe.py         # §4
│   │   └── pipeline.py       # §5
│   ├── data/
│   │   ├── sources.py        # §1 레지스트리
│   │   └── loaders.py        # 다운로드 + 로딩
│   ├── featurize/
│   │   ├── fingerprints.py   # §7.1
│   │   └── descriptors.py    # §7.2
│   ├── splits/
│   │   └── strategies.py     # §6
│   ├── evaluation/
│   │   ├── metrics.py        # §8
│   │   ├── stats.py          # §9
│   │   ├── budget.py         # §10
│   │   └── runner.py         # §11
│   └── models/
│       └── gbm.py            # §7.3
├── scripts/
│   ├── download_data.py
│   ├── curate_dataset.py
│   ├── audit_dataset.py      # §13 — 03 문서 수치 재현
│   └── run_baselines.py
├── tests/
├── docs/                     # 데이터 카드 산출물
└── reports/
```

---

## 3. `curate/standardize.py`

### 3.1 표준화 순서 (고정)

```
1. SMILES 파싱 (sanitize)
2. rdMolStandardize.Cleanup        — 금속 분리, 기본 정규화
3. [조건부] LargestFragmentChooser — 염/짝이온 제거   ← §3.2 참조
4. Uncharger                       — 중성화
5. TautomerEnumerator.Canonicalize — 토토머 정규화
6. 입체화학 정책 적용              — keep | strip
7. 필터 → canonical SMILES + InChIKey
```

순서가 결과를 바꾼다. 3단계를 적용할 경우 반드시 4단계(중성화)보다 **먼저** 해야 한다.

### 3.2 ⚠️ 과제별 염 처리 정책 (감사 결과)

**3단계는 과제에 따라 끄거나 켠다. 기본값은 끈 상태다.**

| 과제 | 다단편(염) 처리 | 근거 |
|---|---|---|
| **용해도 (회귀)** | **`strip_salts=False`. 다단편 입력을 거부한다** | 짝이온이 용해도를 결정한다. 염을 벗기면 화학적으로 다른 물질이 하나로 병합되어 라벨이 최대 9.48 logS까지 충돌한다 |
| 독성 (분류) | `strip_salts=True` 허용 | 활성은 보통 모분자가 결정하므로 염 제거가 통례. **단 Tox21에도 동일 감사를 돌려 확인할 것** |

AqSolDB 9,982건 실측 비교:

| 정책 | 통과 | 고유 분자 | **인공 중복 그룹** | spread>1.0 |
|---|---:|---:|---:|---:|
| `strip_salts=True` | 9,611 | 9,276 | **191** | 102 |
| **`strip_salts=False`** | 8,746 | 8,740 | **6** | **1** |

약 900건(9%)을 잃는 대신 인공 충돌이 사실상 사라진다.

> 예: `4-hydroxybenzenesulfonic acid`(logS 0.76)와 그 소듐염(logS -8.36)은 염 제거 후
> 같은 분자가 된다. 탄산염 21종(탄산나트륨 +0.58 ~ 탄산납 -5.55)도 전부 하나로 병합된다.
> 이것은 노이즈가 아니라 **큐레이션이 만들어낸 오류**다.

### 3.3 설정

```python
class StereoPolicy(str, Enum):
    KEEP = "keep"
    STRIP = "strip"

@dataclass(frozen=True)
class StandardizeConfig:
    stereo: StereoPolicy = StereoPolicy.KEEP
    canonicalize_tautomer: bool = True
    strip_salts: bool = False              # ★ 기본 False. 용해도는 반드시 False
    reject_multi_component: bool = True    # 다단편이면 거부
    organic_only: bool = True
    min_heavy_atoms: int = 3
    max_heavy_atoms: int = 100
    max_mol_weight: float = 1000.0
```

`strip_salts=False`이면 3단계를 건너뛰고, 다단편 입력은 `reject_multi_component`에 의해 `multi_component`로 거부된다.

유기 소분자 원소 화이트리스트:
`H B C N O F Si P S Cl Se Br I`

이 밖의 원소(금속 등)가 남아 있으면 소분자 모델의 적용 대상이 아니므로 거부한다. 탄소가 없는 분자도 거부한다.

### 3.4 반환 타입

```python
@dataclass
class StandardizeResult:
    ok: bool
    smiles: str | None = None        # canonical
    inchikey: str | None = None
    reason: str | None = None        # 실패 사유 (집계용)
    changed: bool = False            # 구조가 바뀌었는지
    notes: list[str] = field(default_factory=list)
```

### 3.5 거부 사유 문자열 (집계 키이므로 고정)

```
empty_input, parse_failed, cleanup_failed, multi_component,
fragment_choice_failed, uncharge_failed,
non_organic_element, no_carbon, too_small, too_large,
molwt_failed, molwt_exceeded, serialization_failed, inchikey_failed
```

`fragment_choice_failed`는 `strip_salts=True`일 때만 발생한다.

### 3.6 구현 주의사항

- **예외를 밖으로 던지지 말 것.** 모든 실패는 `reason`으로 반환해 집계 가능하게 한다
- `RDLogger.DisableLog("rdApp.*")` 로 로그 억제
- `TautomerEnumerator`·`Uncharger`·`LargestFragmentChooser`는 생성 비용이 있다. 클래스 인스턴스에 **한 번만** 만들어 재사용
- 토토머 폭발 방지: `CleanupParameters()`의 `maxTautomers = 200`, `maxTransforms = 200`
- 토토머 정규화 **실패는 치명적이지 않다.** 원본을 유지하고 `notes`에 기록만 남긴다
- **InChIKey 생성 실패는 거부 사유다.** 중복 판정의 근거가 사라지기 때문
- AqSolDB 실측 거부 분포 (`strip_salts=True` 기준, 참고값):
  `no_carbon 190 / non_organic_element 179 / molwt_exceeded 33 / still_multi 43 / too_small 24 / too_large 5 / parse_failed 2`

---

## 4. `curate/dedupe.py`

### 4.1 중복을 두 단계로 본다

| 종류 | 판정 | 처리 |
|---|---|---|
| **완전 중복** | InChIKey 전체 일치 | 병합 또는 거부 |
| **골격 중복** | InChIKey 첫 블록(14자) 일치 | **제거하지 않고 개수만 보고** |

```python
def _skeleton(inchikey: str) -> str:
    return inchikey.split("-")[0]
```

골격 중복은 입체이성질체 등 구조 골격이 같은 분자다. train/test로 갈라지면 누수가 되므로 반드시 **보고**하고, 분할 단계에서 다시 점검한다.

AqSolDB 실측: InChIKey 고유 9,982 / 골격 고유 9,671 → **골격 충돌 311건 규모**

### 4.2 라벨 충돌 처리 — 병합이 아니라 거부가 기본

```python
class ConflictRule(str, Enum):
    DROP = "drop"          # 충돌 건 전부 제거 (회귀 기본값)
    MEDIAN = "median"      # 회귀: 중앙값
    MEAN = "mean"          # 회귀: 평균
    MAJORITY = "majority"  # 분류: 다수결, 동수면 제거
```

| 데이터셋 | tolerance | 기본 규칙 | 근거 |
|---|---:|---|---|
| **AqSolDB** | **0.5** | **DROP** | 염 미제거 정책에서는 잔존 충돌이 6그룹뿐. 병합하지 말고 버린다 |
| ESOL | 0.5 | **DROP** | SMILES에 입체 정보가 없어 만니톨(0.06)/소르비톨(1.09)이 병합된다. MEDIAN은 가짜 라벨 0.575를 만든다 |
| Lipophilicity | 0.5 | MEDIAN | — |
| Tox21 | — | MAJORITY | 분류. `tolerance` 무시, `nunique()>1`이면 충돌 |

`tolerance`는 회귀에서 `max - min`이 이 값을 넘으면 충돌로 보는 임계값이다.

> **충돌은 병합 대상이 아니라 거부 대상이다.** 중앙값으로 뭉개면 화학적 구분이
> 소리 없이 사라진다. 근거: `03_AQSOLDB_AUDIT.md` §4.

### 4.3 AqSolDB 전용 컬럼 보존

`keep_cols`로 반드시 들고 가야 하는 컬럼이다.

| 컬럼 | 용도 |
|---|---|
| `SD` | 출처 간 측정 표준편차 → **노이즈 하한 추정** |
| `Ocurrences` | 출처 개수 → 라벨 신뢰도 |
| `Group` | AqSolDB 자체 신뢰도 등급 G1~G5 → **층화 평가** |

실측 노이즈 하한: 복수 출처 분자 2,236건의 SD 평균 **0.301 logS**.
`LIMITATIONS.md`에 "RMSE 0.3 미만은 측정 노이즈 구간"으로 명기한다.

### 4.4 시그니처

```python
def deduplicate(
    df: pd.DataFrame,
    *,
    key_col: str = "inchikey",
    label_col: str = "label",
    task: Literal["regression", "classification"] = "regression",
    rule: ConflictRule = ConflictRule.DROP,
    tolerance: float = 0.5,
    max_examples: int = 20,
) -> tuple[pd.DataFrame, DedupeReport]: ...
```

`DedupeReport` 필드:

```
n_input, n_output, n_exact_duplicate_rows, n_unique_molecules,
n_conflicting_molecules, n_dropped_by_conflict,
n_skeleton_collision_groups, n_skeleton_collision_molecules,
conflict_rule, conflict_tolerance, examples(최대 20건)
```

### 4.5 누수 점검 (모든 실험 전 필수)

```python
def leakage_check(train_keys, test_keys) -> dict:
    """return {"exact_overlap": int, "skeleton_overlap": int, "clean": bool}"""
```

`exact_overlap > 0`이면 실험을 **중단**시킨다. `skeleton_overlap`은 경고로 보고한다.

---

## 5. `curate/pipeline.py`

### 5.1 `CurationLedger` 추적 항목

```
dataset, task, label_name, label_unit, created_utc
n_raw                                     # 원본
n_after_missing                           # 결측 제거 후
n_after_standardize                       # 표준화 후
n_after_dedupe                            # 중복 제거 후
reject_reasons                            # {사유: 개수}
standardize_notes                         # {변경유형: 개수}
n_structure_changed                       # SMILES가 바뀐 분자 수
n_duplicates_created_by_standardization   # ★ 표준화가 '만들어낸' 중복 그룹 수
duplicate_label_spread_stats              # ★ 그 그룹들의 라벨 불일치 분포
dedupe                                    # DedupeReport dict
standardize_config                        # 설정 전체
label_stats                               # 라벨 분포
```

★ 두 항목이 이번 감사로 추가됐다. **들어온 중복과 내가 만든 중복을 구분해 보고하는 것**이 이 데이터 카드의 핵심이다.

### 5.2 라벨 분포 통계

- 회귀: `n, mean, std, min, p25, median, p75, max`
- 분류: `n, n_positive, n_negative, positive_rate, imbalance_ratio`

**`imbalance_ratio > 5`이면 데이터 카드에 경고를 자동 삽입한다.**
"음성:양성 = N:1 로 불균형이 심하므로 AUROC 단독 보고는 낙관적이다. AUPRC와 PPV/NPV를 병기한다."

### 5.3 `render_data_card(ledger) -> str`

마크다운 섹션 구성:

1. 단계별 생존 개수 (표 + 직전 대비 증감 + 최종 보존율 %)
2. 표준화 거부 사유 (개수 내림차순)
3. 구조 변경 내역
4. 중복 및 라벨 충돌 (골격 충돌 누수 경고 포함)
5. **표준화가 만들어낸 중복** — 들어온 중복과 내가 만든 중복을 구분 ★
6. 라벨 분포 (불균형 경고 자동)
7. 표준화 설정 (JSON) + 순서 고정 안내문

> **이 데이터 카드가 1주차의 핵심 산출물이다.** 모델보다 중요하다.
> "데이터 몇 개 썼나요?" → "원본 9,982, 표준화 후 8,746, 최종 8,740.
> 염을 벗기지 않은 이유는…"이 나와야 한다.

---

## 6. `splits/strategies.py`

### 6.1 ⚠️ 가장 중요한 함정

```
scaffold        → 결정론적. 시드를 10개 돌려도 분산이 0이다.
scaffold_random → 골격 그룹을 시드로 섞는다. 시드 분산 측정은 반드시 이쪽.
```

`iter_splits`는 `scaffold`와 `time`에 대해 **시드 루프를 1회로 축약**해야 한다.

### 6.2 시그니처

```python
@dataclass(frozen=True)
class Split:
    train: np.ndarray   # 위치 인덱스 (0-based)
    valid: np.ndarray
    test:  np.ndarray
    strategy: str
    seed: int
    meta: dict

def make_split(
    df, *, strategy: SplitName, seed: int,
    smiles_col="smiles", date_col=None,
    frac_train=0.8, frac_valid=0.1, frac_test=0.1,
) -> Split: ...

def iter_splits(df, *, strategies, seeds, **kwargs): ...
```

### 6.3 골격 분할 구현

- `MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)`
- 골격 추출 실패분은 **각자 단독 그룹**으로 둔다 (`f"__unparsed_{i}__"`). 임의로 하나로 병합하면 거대 그룹이 생겨 분할이 망가진다
- 그룹 할당: train 목표치까지 채우고 → valid → 나머지 test
- 결정론적 모드의 정렬 키: `(-len(group), scaffold_smiles)` — 동일 크기일 때도 재현되도록 문자열로 안정 정렬
- `meta`에 `n_scaffold_groups`, `n_scaffold_parse_failed`, `largest_group` 기록

### 6.4 시간 분할 구현

- 날짜 **결측을 허용하지 않는다.** 결측이 있으면 예외를 던지고 건수를 알린다
- 경계에 같은 값(같은 연도)이 걸치면 그 값이 끝나는 곳까지 경계를 **민다**
- ESOL / Tox21 / AqSolDB / Lipophilicity 모두 날짜 컬럼이 **없다**. `date_col` 없이 호출되면 명확한 안내와 함께 예외를 던진다 (ChEMBL 추출분에만 적용)
- `meta`에 `train_range`, `test_range` 기록

---

## 7. 피처와 모델 (3주차)

### 7.1 `featurize/fingerprints.py`

- ECFP4 = Morgan, `radius=2`, `nBits=2048`
- 권장 API: `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)`
  (`GetMorganFingerprintAsBitVect`는 deprecated 경고가 난다)
- 카운트 버전도 옵션으로 둔다 (`GetCountFingerprint`)

### 7.2 `featurize/descriptors.py`

- `Descriptors.CalcMolDescriptors(mol)` → 전체 2D 서술자 dict
- **NaN/inf 처리 필수.** 일부 서술자는 특정 분자에서 발산한다
- 학습셋에서 상수 컬럼(분산 0)과 결측 과다 컬럼을 제거하고, 그 **컬럼 목록을 고정**해 valid/test에 동일 적용 (분산 누수 방지)
- 사용한 서술자 개수와 제거된 컬럼 수를 리포트에 기록
- AqSolDB에는 RDKit 서술자 17종이 이미 들어 있다. **직접 계산한 값과 일치하는지 한 번 대조**해 두면 좋다 (버전 차이 확인)

### 7.3 `models/gbm.py`

| ID | 피처 |
|---|---|
| A | ECFP4 2048 |
| B | RDKit 2D 서술자 |
| A+B | 두 개 결합 |

**공정성 핵심 — 동일 탐색 예산**

```python
def random_search(
    model_fn, param_space, X_tr, y_tr, X_va, y_va,
    *, n_trials: int, seed: int, metric: str,
) -> tuple[dict, float]:
    """모든 모델이 같은 n_trials를 쓴다. 이 숫자를 리포트에 명시한다."""
```

- 탐색은 **valid**에서만 평가
- `n_trials`를 모델별로 다르게 주면 안 된다. GNN만 100회 돌리고 LightGBM은 기본값으로 돌린 비교는 비교가 아니다
- 선택된 하이퍼파라미터를 결과 테이블에 함께 저장
- 라벨 예산이 작을 때(100~300) LightGBM 기본값은 과적합한다. `min_child_samples`·`num_leaves`를 탐색 공간에 반드시 포함

---

## 8. `evaluation/metrics.py`

### 회귀

```
RMSE, MAE, Spearman ρ, Pearson r
```

- **Pearson r을 제곱해 R²로 쓰지 말 것.** 둘은 다른 값이다. R²가 필요하면 `1 - SS_res/SS_tot`로 직접 계산
- 주 지표는 **RMSE**. MAE는 RMSE를 과소평가하므로 병기
- AqSolDB 기준 **RMSE 0.3 미만은 측정 노이즈 구간**임을 결과 해석에 반영

### 분류

```
AUPRC (average_precision), AUROC, PPV, NPV, MCC, balanced accuracy
```

- **주 지표는 AUPRC.** Tox21은 불균형이 심해 AUROC가 낙관적으로 보인다
- PPV/NPV의 임계값은 **valid에서 결정**하고 test에 그대로 적용. test에서 임계값을 고르면 누수다
- AUROC도 함께 보고하되, 불균형 비율을 같은 표에 적는다
- **테스트셋 양성 개수를 결과 테이블에 항상 기록한다.** 양성이 20개 미만인 조합은 신뢰할 수 없음을 표시

---

## 9. `evaluation/stats.py`

```python
def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    """시드별 쌍 비교. return {statistic, p_value, median_diff, n_pairs}"""
```

- `scipy.stats.wilcoxon`, 시드별로 쌍을 맞춘다
- 시드 수가 5개 미만이면 검정력이 없다. **경고를 띄울 것**
- 평균 차이만 보고하지 말고 **중앙값 차이와 p값**을 함께
- 다중 비교를 할 경우 보정 여부를 명시 (Holm 권장)

---

## 10. `evaluation/budget.py`

```python
def subsample_train(
    y_train, *, n_labels: int, seed: int, task: str, stratify: bool = True
) -> np.ndarray:
    """학습셋 인덱스를 n_labels개로 줄인다. 테스트셋은 절대 건드리지 않는다."""
```

규칙:

- 분류는 **양성 비율을 유지한 층화 샘플링**
- `n_labels`가 학습셋보다 크면 전체를 쓰고 그 사실을 기록
- **서브샘플링 시드와 분할 시드를 분리한다.** 같은 시드를 쓰면 두 효과가 뒤섞여 분산 해석이 불가능해진다
- 예산 사다리 (§1.4 근거):

```python
BUDGETS_REGRESSION     = [100, 300, 1000, 3000, None]   # None = 전체
BUDGETS_CLASSIFICATION = [300, 1000, 3000, None]         # 100 제외
```

- 분류에서 서브샘플 결과의 **양성이 20개 미만이면 경고를 띄우고 그 조합을 결과에 플래그**한다

---

## 11. `evaluation/runner.py`

(모델 × 분할 × 예산 × 시드) 를 순회하며 tidy DataFrame을 반환한다.

필수 출력 컬럼:

```
dataset, task, target_task, model, split_strategy, split_seed,
n_labels, subsample_seed, metric_name, metric_value,
n_train, n_valid, n_test, n_test_positive,
hp_n_trials, hp_selected(json),
leakage_exact, leakage_skeleton, low_positive_flag, runtime_sec
```

- 결과는 매 조합마다 **즉시 append 저장**(CSV/parquet). 중간에 죽어도 재시작 가능해야 한다
- 이미 계산된 조합은 건너뛰는 resume 로직
- 각 조합 시작 시 `leakage_check`를 실행해 결과 컬럼에 남긴다

---

## 12. 수락 기준 (테스트)

1주차 완료 판정은 아래 통과로 한다.

### 표준화

| 테스트 | 검증 내용 |
|---|---|
| `test_reject_multi_component_by_default` | **기본 설정에서 염 포함 SMILES가 `multi_component`로 거부되는지** |
| `test_strip_salts_when_enabled` | `strip_salts=True`일 때만 짝이온이 제거되는지 |
| `test_standardize_charge` | 하전 분자가 중성화되는지 |
| `test_standardize_tautomer` | 같은 분자의 다른 토토머 표기가 **같은 canonical SMILES**로 수렴하는지 |
| `test_standardize_reject_metal` | 금속 포함 분자가 `non_organic_element`로 거부되는지 |
| `test_standardize_no_exception` | 깨진 SMILES 100종에 대해 예외를 던지지 않는지 |

### 중복·충돌

| 테스트 | 검증 내용 |
|---|---|
| `test_dedupe_exact` | 완전 중복이 1건으로 합쳐지는지 |
| `test_dedupe_conflict_drop` | tolerance 초과 충돌이 DROP 규칙에서 제거되고 집계되는지 |
| `test_dedupe_skeleton_report` | 입체이성질체 쌍이 골격 충돌로 **보고**되지만 제거되지 않는지 |
| `test_salt_merge_regression` | **염 제거를 켜면 탄산나트륨/탄산칼슘이 같은 InChIKey로 병합됨을 명시적으로 검증** (정책 근거의 회귀 테스트) |

### 분할·예산

| 테스트 | 검증 내용 |
|---|---|
| `test_split_scaffold_deterministic` | `scaffold`가 시드와 무관하게 동일 결과인지 |
| `test_split_scaffold_random_varies` | `scaffold_random`이 시드에 따라 달라지는지 |
| `test_split_no_leakage` | 모든 분할에서 `exact_overlap == 0`인지 |
| `test_split_time_boundary` | 같은 연도가 train/test에 걸치지 않는지 |
| `test_split_time_requires_date` | 날짜 컬럼 없이 호출하면 예외가 나는지 |
| `test_budget_stratify` | 층화 서브샘플링이 양성 비율을 유지하는지 |
| `test_budget_seed_independence` | 분할 시드 고정 + 서브샘플 시드 변경 시 테스트셋이 불변인지 |
| `test_budget_low_positive_warning` | 양성 20개 미만 조합에 플래그가 붙는지 |

### 통합

| 테스트 | 검증 내용 |
|---|---|
| `test_e2e_esol` | ESOL 큐레이션 → 데이터 카드 생성 성공 |
| `test_e2e_tox21_sr_are` | Tox21 SR-ARE 큐레이션 → 데이터 카드 생성 성공 |
| `test_audit_reproduces_numbers` | **`scripts/audit_dataset.py`가 `03` 문서의 정책 A/B 비교 수치를 재현하는지** |

---

## 13. `scripts/audit_dataset.py`

`03_AQSOLDB_AUDIT.md`의 수치를 코드로 재현하는 스크립트다. 문서의 숫자가 손으로 적힌 값이 아니라 **재현 가능한 산출물**이어야 한다.

출력해야 할 항목:

1. 파일 shape, 컬럼 목록
2. 라벨 분포 통계
3. `SD` / `Ocurrences` / `Group` 분포 (해당 컬럼이 있는 데이터셋만)
4. 원본 InChIKey 고유 수 / 골격 고유 수
5. **정책 A(`strip_salts=True`) vs 정책 B(`strip_salts=False`) 비교표**
   — 통과 건수, 고유 분자, 인공 중복 그룹 수, spread 분포
6. 가장 심한 충돌 상위 N건 (분자명과 라벨 포함)
7. 측정 노이즈 하한 추정값

```
python scripts/audit_dataset.py --dataset aqsoldb --out docs/audit_aqsoldb.md
python scripts/audit_dataset.py --dataset tox21 --task SR-ARE --out docs/audit_tox21.md
```

**Tox21에도 반드시 돌려서** 염 처리 정책을 데이터로 결정한다. 용해도와 결론이 다를 수 있다.

---

## 14. 1주차 작업 순서

1. 스캐폴드 + `requirements.txt` + `pyproject.toml`
2. `data/sources.py` 레지스트리 (§1 표 그대로)
3. `data/loaders.py` — 다운로드 캐시, `DatasetUnavailable` 예외
4. `curate/standardize.py` + 테스트 6종 (§12)
5. `curate/dedupe.py` + 테스트 4종 (§12)
6. `curate/pipeline.py` + `render_data_card`
7. `scripts/curate_dataset.py` — CLI (`--dataset aqsoldb`, `--dataset tox21 --task SR-ARE`)
8. `scripts/audit_dataset.py` — §13
9. AqSolDB / ESOL / Tox21(SR-ARE) 큐레이션 실행 → `docs/data_card_*.md` 커밋
10. **Tox21 감사 실행 → 염 처리 정책 결정 및 스펙 반영**

2주차는 §6·§8·§9·§10, 3주차는 §7·§11로 이어진다.

---

## 15. 변경 이력

| 날짜 | 변경 | 근거 |
|---|---|---|
| 2026-10-22 | `strip_salts` 기본값 False로 설정. 용해도 과제에서 염 제거 금지 | `03` §4 — 인공 중복 191→6 그룹 |
| 2026-10-22 | AqSolDB 충돌 규칙을 MEDIAN(tolerance 1.0) → **DROP(tolerance 0.5)** | `03` §4 — 중앙값 병합이 화학 정보를 파괴 |
| 2026-10-22 | `SD`·`Ocurrences`·`Group` 컬럼 보존 의무화, 층화 평가 추가 | `03` §6·§7 |
| 2026-10-22 | 원장에 `n_duplicates_created_by_standardization` 추가 | `03` §4 |
| 2026-10-22 | 분류 주 과제를 `NR-AR` → **`SR-ARE`**, 분류 예산에서 100 제외 | §1.4 — 양성 개수 부족 |
| 2026-10-22 | Tox21·Lipophilicity의 `Unnamed: 0` 제거 명시 | §1.3 — 실측 확인 |
| 2026-10-22 | ESOL `predicted` 컬럼 사용 금지 명시 | §1.3 — 다른 모델의 출력 |
| 2026-10-22 | 노이즈 하한 0.301 logS 기재 | `03` §6 |
| 2026-10-22 | `scripts/audit_dataset.py` 신설 | §13 |
| 2026-10-22 | `stereo=keep`일 때 토토머 정규화의 입체 제거(`tautomerRemoveSp3Stereo`/`BondStereo`)를 끈다 | RDKit 기본값이 L/D-알라닌을 병합함을 실측 |
| 2026-10-22 | 토토머 정규화 유지 + 원장에 `n_duplicates_incoming`, `n_duplicates_created_by_tautomer` 추가, 데이터 카드 §5에서 발생 단계별 집계 | 토토머 포함 시 AqSolDB 중복 6→26그룹 (`03` 상단 주석) |
| 2026-10-22 | 03 감사 수치는 토토머 정규화 없이 측정된 값임을 명기 | 재측정 |
| 2026-10-22 | ESOL 충돌 규칙 MEDIAN → **DROP** (최종 1,110 → 1,109) | 입체 정보 없는 SMILES에서 만니톨/소르비톨 병합, 중앙값이 실측에 없는 라벨 생성 |
