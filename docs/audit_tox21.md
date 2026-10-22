# 감사 보고서 — tox21

> `scripts/audit_dataset.py`가 생성. 문서와 숫자가 어긋나면 이 출력을 신뢰한다.

## 1. 파일

- shape: 7,831 × 13
- 컬럼: `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`, `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`, `smiles`

## 2. 라벨 분포 (`SR-ARE`)

결측 라벨: 1,999

| 통계 | 값 |
|---|---:|
| n | 5,832 |
| n_positive | 942 |
| n_negative | 4,890 |
| positive_rate | 0.162 |
| imbalance_ratio | 5.191 |

## 3. 신뢰도 컬럼

해당 컬럼 없음.

## 4. 원본 InChIKey

| 항목 | 값 |
|---|---:|
| n_parsed | 7,823 |
| n_unique_inchikey | 7,821 |
| n_unique_skeleton | 7,619 |

## 5. 염 처리 정책 비교

A = 염 제거(`strip_salts=True`), B = 다단편 거부(`strip_salts=False`). `no_taut`는 토토머 정규화 미포함(03 문서 측정 조건).

| 지표 | `A_strip` | `B_reject` | `A_strip_no_taut` | `B_reject_no_taut` |
|---|---:|---:|---:|---:|
| 통과 | 5,715 | 5,576 | 5,715 | 5,576 |
| 고유 InChIKey | 5,569 | 5,517 | 5,573 | 5,521 |
| 중복 그룹 (전체) | 104 | 59 | 100 | 55 |
| 표준화가 만든 중복 그룹 | 102 | 57 | 98 | 53 |
| └ 관련 행 | 246 | 114 | 238 | 106 |
| 만든 중복 spread n_label_disagree | 16 | 3 | 16 | 3 |

거부 사유:

- `A_strip`: non_organic_element 56, no_carbon 34, molwt_exceeded 14, parse_failed 7, too_small 4, too_large 1, inchikey_failed 1
- `B_reject`: multi_component 171, non_organic_element 40, no_carbon 21, molwt_exceeded 13, parse_failed 7, too_small 2, too_large 1, inchikey_failed 1
- `A_strip_no_taut`: non_organic_element 56, no_carbon 34, molwt_exceeded 14, parse_failed 7, too_small 4, too_large 1, inchikey_failed 1
- `B_reject_no_taut`: multi_component 171, non_organic_element 40, no_carbon 21, molwt_exceeded 13, parse_failed 7, too_small 2, too_large 1, inchikey_failed 1

## 6. 표준화가 만든 중복 — 상위 예시

### `A_strip`

- spread **1.00** (n=3)
    -    1.00  row2119  `CCCCCC[N+]1:C:C:C:C:C:1`
    -    1.00  row3057  `CCCCCC[N+]1:C:C:C:C:C:1.O=S(=O)([O-])C(F)(F)F`
    -    0.00  row3040  `CCCCCC[N+]1:C:C:C:C:C:1.F[B-](F)(F)F`
- spread **1.00** (n=2)
    -    1.00  row6478  `CN1:C(=O):C2:C(:N:C:N:2C):N(C):C:1=O`
    -    0.00  row4700  `CN1:C(=O):C2:C(:N:C:N:2C):N(C):C:1=O.O=C(O)CC(O)(CC(=O)O)C(=O)O`
- spread **1.00** (n=2)
    -    1.00  row7501  `[O-]C1:C:C:C:C:C:1C1:C:C:C:C:C:1`
    -    0.00  row1988  `OC1:C:C:C:C:C:1C1:C:C:C:C:C:1`
- spread **1.00** (n=4)
    -    1.00  row7641  `CCC(C)(CCC(C)C)C(=O)O[Sn](C)(C)OC(=O)C(C)(CC)CCC(C)C`
    -    0.00  row1890  `CCC(C)(CCC(C)C)C(=O)O`
    -    0.00  row3648  `CCC(C)(CCC(C)C)C(=O)[O-].CCC(C)(CCC(C)C)C(=O)[O-].[Zn+2]`
    -    0.00  row3913  `CCC(C)(CCC(C)C)C(=O)[O-].CCC(C)(CCC(C)C)C(=O)[O-]`
- spread **1.00** (n=2)
    -    1.00  row2977  `CC(=O)C(=O)O`
    -    0.00  row2377  `CC(=O)C(=O)[O-]`
- spread **1.00** (n=2)
    -    1.00  row2384  `CCCC[N+]1:C:C:C(C):C:C:1`
    -    0.00  row7152  `CCCC[N+]1:C:C:C(C):C:C:1.O=S(=O)([O-])C(F)(F)F`
- spread **1.00** (n=3)
    -    1.00  row1728  `CCCC[N+]1:C:C:C:C(C):C:1`
    -    0.00  row6782  `CCCC[N+]1:C:C:C:C(C):C:1.F[B-](F)(F)F`
    -    0.00  row7155  `CCCC[N+]1:C:C:C:C(C):C:1.O=S(=O)([O-])C(F)(F)F`
- spread **1.00** (n=2)
    -    1.00  row2630  `CC[N+]1:C:C:C:C:C:1`
    -    0.00  row7319  `CC[N+]1:C:C:C:C:C:1.F[B-](F)(F)F`
- spread **1.00** (n=3)
    -    1.00  row5338  `CCCCCCCCCC(CC)C1:C:C:C(S(=O)(=O)[O-]):C:C:1`
    -    1.00  row5649  `CCCCCCCCCC(CC)C1:C:C:C(S(=O)(=O)O):C:C:1`
    -    0.00  row1603  `CCCCCCCCCC(CC)C1:C:C:C(S(=O)(=O)O):C:C:1.OCCN(CCO)CCO`
- spread **1.00** (n=3)
    -    1.00  row3324  `CCCCCCCCCCCCN1:C:C:[N+](C):C:1.F[B-](F)(F)F`
    -    1.00  row7413  `CCCCCCCCCCCCN1:C:C:[N+](C):C:1.O=S(=O)([O-])C(F)(F)F`
    -    0.00  row3445  `CCCCCCCCCCCCN1:C:C:[N+](C):C:1`

### `B_reject`

- spread **1.00** (n=2)
    -    1.00  row4531  `CCCCCCCCCCCC(=O)N(C)CC(=O)O`
    -    0.00  row3729  `CCCCCCCCCCCC(=O)N(C)CC(=O)[O-]`
- spread **1.00** (n=2)
    -    1.00  row7501  `[O-]C1:C:C:C:C:C:1C1:C:C:C:C:C:1`
    -    0.00  row1988  `OC1:C:C:C:C:C:1C1:C:C:C:C:C:1`
- spread **1.00** (n=2)
    -    1.00  row2977  `CC(=O)C(=O)O`
    -    0.00  row2377  `CC(=O)C(=O)[O-]`
- spread **0.00** (n=2)
    -    0.00  row4  `CC(O)(P(=O)(O)O)P(=O)(O)O`
    -    0.00  row7497  `CC(O)(P(=O)([O-])O)P(=O)([O-])O`
- spread **0.00** (n=2)
    -    0.00  row2362  `O=C(O)C1=NN(C2:C:C:C(S(=O)(=O)O):C:C:2)C(=O)C1`
    -    0.00  row3285  `O=C([O-])C1=NN(C2:C:C:C(S(=O)(=O)O):C:C:2)C(=O)C1`
- spread **0.00** (n=2)
    -    0.00  row2531  `O=C(O)C(Cl)(Cl)Cl`
    -    0.00  row6179  `O=C([O-])C(Cl)(Cl)Cl`
- spread **0.00** (n=2)
    -    0.00  row3115  `O=C(O)C1:C:C:C:C:C:1C(=O)O`
    -    0.00  row5976  `O=C([O-])C1:C:C:C:C:C:1C(=O)O`
- spread **0.00** (n=2)
    -    0.00  row3122  `O=C1O[C@H]([C@H](O)CO)C(O)=C1O`
    -    0.00  row4060  `O=C1O[C@H]([C@H](O)CO)C([O-])=C1O`
- spread **0.00** (n=2)
    -    1.00  row3124  `COC1:C:C:N:C(CS(=O)C2:N:C3:C:C(OC(F)F):C:C:C:3:[N-]:2):C:1OC`
    -    1.00  row3278  `COC1:C:C:N:C(CS(=O)C2:N:C3:C:C(OC(F)F):C:C:C:3:[NH]:2):C:1OC`
- spread **0.00** (n=2)
    -    0.00  row3167  `CC1:C:C(Cl):C:C:C:1OCC(=O)[O-]`
    -    0.00  row6828  `CC1:C:C(Cl):C:C:C:1OCC(=O)O`

### `A_strip_no_taut`

- spread **1.00** (n=3)
    -    1.00  row1728  `CCCC[N+]1:C:C:C:C(C):C:1`
    -    0.00  row6782  `CCCC[N+]1:C:C:C:C(C):C:1.F[B-](F)(F)F`
    -    0.00  row7155  `CCCC[N+]1:C:C:C:C(C):C:1.O=S(=O)([O-])C(F)(F)F`
- spread **1.00** (n=3)
    -    1.00  row4896  `OC1:C:C:C:C2:C:C:C:N:C:1:2`
    -    1.00  row5693  `OC1:C:C:C:C2:C:C:C:N:C:1:2.OC1:C:C:C:C2:C:C:C:N:C:1:2`
    -    0.00  row344  `O=C(O)C1:C:C:C:C:C:1O.OC1:C:C:C:C2:C:C:C:N:C:1:2`
- spread **1.00** (n=2)
    -    1.00  row4531  `CCCCCCCCCCCC(=O)N(C)CC(=O)O`
    -    0.00  row3729  `CCCCCCCCCCCC(=O)N(C)CC(=O)[O-]`
- spread **1.00** (n=7)
    -    1.00  row1640  `Cl[V](Cl)(C1C=CC=C1)C1C=CC=C1`
    -    1.00  row5946  `C1:C:C:[CH-]:C:1.C1:C:C:[CH-]:C:1.[Co+2]`
    -    0.00  row988  `C1:C:C:[CH-]:C:1.C1:C:C:[CH-]:C:1.[Fe+2]`
    -    0.00  row2501  `C1:C:C:[CH-]:C:1.C1:C:C:[CH-]:C:1.[Ni+2]`
    -    0.00  row2851  `C1=CCC=C1`
    -    0.00  row3390  `Cl[Zr](Cl)(C1=CC=CC1)C1C=CC=C1`
    - … 외 1건
- spread **1.00** (n=3)
    -    1.00  row5338  `CCCCCCCCCC(CC)C1:C:C:C(S(=O)(=O)[O-]):C:C:1`
    -    1.00  row5649  `CCCCCCCCCC(CC)C1:C:C:C(S(=O)(=O)O):C:C:1`
    -    0.00  row1603  `CCCCCCCCCC(CC)C1:C:C:C(S(=O)(=O)O):C:C:1.OCCN(CCO)CCO`
- spread **1.00** (n=2)
    -    1.00  row6746  `O=C([O-])CN(CCN(CC(=O)[O-])CC(=O)[O-])CC(=O)[O-].[Fe+3]`
    -    0.00  row3980  `O=C(O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O`
- spread **1.00** (n=2)
    -    1.00  row6478  `CN1:C(=O):C2:C(:N:C:N:2C):N(C):C:1=O`
    -    0.00  row4700  `CN1:C(=O):C2:C(:N:C:N:2C):N(C):C:1=O.O=C(O)CC(O)(CC(=O)O)C(=O)O`
- spread **1.00** (n=3)
    -    1.00  row3324  `CCCCCCCCCCCCN1:C:C:[N+](C):C:1.F[B-](F)(F)F`
    -    1.00  row7413  `CCCCCCCCCCCCN1:C:C:[N+](C):C:1.O=S(=O)([O-])C(F)(F)F`
    -    0.00  row3445  `CCCCCCCCCCCCN1:C:C:[N+](C):C:1`
- spread **1.00** (n=2)
    -    1.00  row2630  `CC[N+]1:C:C:C:C:C:1`
    -    0.00  row7319  `CC[N+]1:C:C:C:C:C:1.F[B-](F)(F)F`
- spread **1.00** (n=2)
    -    1.00  row3427  `CCCCCCCCCCCC(=O)O[Sn](CCCC)(CCCC)OC(=O)CCCCCCCCCCC`
    -    0.00  row6971  `CCCCCCCCCCCC(=O)O`

### `B_reject_no_taut`

- spread **1.00** (n=2)
    -    1.00  row2977  `CC(=O)C(=O)O`
    -    0.00  row2377  `CC(=O)C(=O)[O-]`
- spread **1.00** (n=2)
    -    1.00  row7501  `[O-]C1:C:C:C:C:C:1C1:C:C:C:C:C:1`
    -    0.00  row1988  `OC1:C:C:C:C:C:1C1:C:C:C:C:C:1`
- spread **1.00** (n=2)
    -    1.00  row4531  `CCCCCCCCCCCC(=O)N(C)CC(=O)O`
    -    0.00  row3729  `CCCCCCCCCCCC(=O)N(C)CC(=O)[O-]`
- spread **0.00** (n=2)
    -    0.00  row3784  `O=C([O-])C1:C:C:C:C:C:1`
    -    0.00  row4225  `O=C(O)C1:C:C:C:C:C:1`
- spread **0.00** (n=2)
    -    0.00  row2531  `O=C(O)C(Cl)(Cl)Cl`
    -    0.00  row6179  `O=C([O-])C(Cl)(Cl)Cl`
- spread **0.00** (n=2)
    -    0.00  row3115  `O=C(O)C1:C:C:C:C:C:1C(=O)O`
    -    0.00  row5976  `O=C([O-])C1:C:C:C:C:C:1C(=O)O`
- spread **0.00** (n=2)
    -    0.00  row3122  `O=C1O[C@H]([C@H](O)CO)C(O)=C1O`
    -    0.00  row4060  `O=C1O[C@H]([C@H](O)CO)C([O-])=C1O`
- spread **0.00** (n=2)
    -    1.00  row3124  `COC1:C:C:N:C(CS(=O)C2:N:C3:C:C(OC(F)F):C:C:C:3:[N-]:2):C:1OC`
    -    1.00  row3278  `COC1:C:C:N:C(CS(=O)C2:N:C3:C:C(OC(F)F):C:C:C:3:[NH]:2):C:1OC`
- spread **0.00** (n=2)
    -    0.00  row3167  `CC1:C:C(Cl):C:C:C:1OCC(=O)[O-]`
    -    0.00  row6828  `CC1:C:C(Cl):C:C:C:1OCC(=O)O`
- spread **0.00** (n=2)
    -    0.00  row3215  `C=C(C)C(=O)O`
    -    0.00  row5420  `C=C(C)C(=O)[O-]`

## 7. 측정 노이즈 하한

`SD` 컬럼 없음 — 추정 불가.
