# 감사 보고서 — aqsoldb

> `scripts/audit_dataset.py`가 생성. 문서와 숫자가 어긋나면 이 출력을 신뢰한다.

## 1. 파일

- shape: 9,982 × 26
- 컬럼: `ID`, `Name`, `InChI`, `InChIKey`, `SMILES`, `Solubility`, `SD`, `Ocurrences`, `Group`, `MolWt`, `MolLogP`, `MolMR`, `HeavyAtomCount`, `NumHAcceptors`, `NumHDonors`, `NumHeteroatoms`, `NumRotatableBonds`, `NumValenceElectrons`, `NumAromaticRings`, `NumSaturatedRings`, `NumAliphaticRings`, `RingCount`, `TPSA`, `LabuteASA`, `BalabanJ`, `BertzCT`

## 2. 라벨 분포 (`Solubility`)

결측 라벨: 0

| 통계 | 값 |
|---|---:|
| n | 9,982 |
| mean | -2.890 |
| std | 2.368 |
| min | -13.172 |
| p25 | -4.326 |
| median | -2.618 |
| p75 | -1.210 |
| max | 2.138 |

## 3. 신뢰도 컬럼

`ocurrences` 분포: 1: 7,746 / 2: 1,417 / 3: 494 / 4: 177 / 5: 81 / 6: 25 / 7: 11 / 8: 14 / 9: 4 / 10: 3 / 12: 2 / 13: 3 / 16: 3 / 27: 1 / 38: 1

`group` 분포: G1: 7,746 / G2: 235 / G3: 1,182 / G4: 183 / G5: 636

## 4. 원본 InChIKey

| 항목 | 값 |
|---|---:|
| n_parsed | 9,980 |
| n_unique_inchikey | 9,980 |
| n_unique_skeleton | 9,669 |
| file_inchikey_unique | 9,982 |
| file_skeleton_unique | 9,671 |

## 5. 염 처리 정책 비교

A = 염 제거(`strip_salts=True`), B = 다단편 거부(`strip_salts=False`). `no_taut`는 토토머 정규화 미포함(03 문서 측정 조건).

| 지표 | `A_strip` | `B_reject` | `A_strip_no_taut` | `B_reject_no_taut` |
|---|---:|---:|---:|---:|
| 통과 | 9,554 | 8,735 | 9,554 | 8,735 |
| 고유 InChIKey | 9,200 | 8,709 | 9,225 | 8,729 |
| 중복 그룹 (전체) | 208 | 26 | 186 | 6 |
| 표준화가 만든 중복 그룹 | 208 | 26 | 186 | 6 |
| └ 관련 행 | 562 | 52 | 515 | 12 |
| 만든 중복 spread median | 0.981 | 0.093 | 1.108 | 0.062 |
| 만든 중복 spread max | 9.481 | 1.723 | 9.481 | 1.350 |
| 만든 중복 spread n_gt_0.5 | 136 | 6 | 129 | 1 |
| 만든 중복 spread n_gt_1.0 | 104 | 4 | 99 | 1 |
| 만든 중복 spread n_gt_2.0 | 59 | 0 | 59 | 0 |

거부 사유:

- `A_strip`: no_carbon 194, non_organic_element 168, molwt_exceeded 35, too_small 24, too_large 5, parse_failed 2
- `B_reject`: multi_component 1,141, non_organic_element 53, no_carbon 25, too_small 14, molwt_exceeded 11, parse_failed 2, too_large 1
- `A_strip_no_taut`: no_carbon 194, non_organic_element 168, molwt_exceeded 35, too_small 24, too_large 5, parse_failed 2
- `B_reject_no_taut`: multi_component 1,141, non_organic_element 53, no_carbon 25, too_small 14, molwt_exceeded 11, parse_failed 2, too_large 1

## 6. 표준화가 만든 중복 — 상위 예시

### `A_strip`

- spread **9.48** (n=11)
    -    0.78  sodium 2-ethylhexanoate  `[Na+].CCCCC(CC)C([O-])=O`
    -    0.46  zinc(2+) ion bis(2-ethylhexanoate)  `[Zn++].CCCCC(CC)C([O-])=O.CCCCC(CC)C([O-])=O`
    -   -0.91  barium(2+) ion bis(2-ethylhexanoate)  `[Ba++].CCCCC(CC)C([O-])=O.CCCCC(CC)C([O-])=O`
    -   -1.59  calcium bis(2-ethylhexanoate)  `[Ca++].CCCCC(CC)C([O-])=O.CCCCC(CC)C([O-])=O`
    -   -1.83  ({bis[(2-ethylhexanoyl)oxy]-2,4-dioxa-3-bora-1,5-dicobaltapentan-3-yl}oxy)cobaltio 2-ethylhexanoate  `CCCCC(CC)C(=O)O[Co]OB(O[Co])O[Co](OC(=O)C(CC)CCCC)OC(=O)C(CC)CCCC`
    -   -1.86  2-ethylhexanoic acid  `CCCCC(CC)C(O)=O`
    - … 외 5건
- spread **9.12** (n=2)
    -    0.76  4-hydroxybenzenesulfonic acid  `Oc1ccc(cc1)[S](O)(=O)=O`
    -   -8.36  sodium 4-hydroxybenzene-1-sulfonate  `[Na+].Oc1ccc(cc1)[S]([O-])(=O)=O`
- spread **7.62** (n=6)
    -    0.19  pentane-2,4-dione  `CC(=O)CC(C)=O`
    -   -1.27  calcium bis((2Z)-4-oxopent-2-en-2-olate)  `[Ca++].CC(=O)[CH-]C(C)=O.CC(=O)[CH-]C(C)=O`
    -   -1.70  cobalt(2+) bis(4-oxopent-2-en-2-olate)  `[Co++].CC([O-])=CC(C)=O.CC([O-])=CC(C)=O`
    -   -2.03  aluminum tris(4-oxopent-2-en-2-olate)  `[Al+3].CC([O-])=CC(C)=O.CC([O-])=CC(C)=O.CC([O-])=CC(C)=O`
    -   -2.07  bis(propan-2-yloxy)titaniumbis(ylium) bis((2Z)-4-oxopent-2-en-2-olate)  `[Ti+4].CC(C)[O-].CC(C)[O-].CC(=O)[CH-]C(C)=O.CC(=O)[CH-]C(C)=O`
    -   -7.44  palladium (II) di(4-oxopent-2-en-2-oate)  `[Pd++].CC(=O)[CH-]C(C)=O.CC(=O)[CH-]C(C)=O`
- spread **7.59** (n=4)
    -   -4.41  octadec-9-enoic acid - N-(2-aminoethyl)ethane-1,2-diamine (1:1)  `CCCCCCCC\C=C/CCCCCCCC(O)=O.NCCNCCN`
    -   -4.81  9-Octadecenoic acid (Z)-, reaction products with 2-[(2-aminoethyl)amino]ethanol  `CCCCCCCC\C=C/CCCCCCCC(O)=O.NCCNCCO`
    -   -6.80  lithium octadec-9-enoate  `[Li+].CCCCCCCC\C=C/CCCCCCCC([O-])=O`
    -  -12.00  copper(2+) ion bis((9Z)-octadec-9-enoate)  `[Cu++].CCCCCCCC\C=C/CCCCCCCC([O-])=O.CCCCCCCC\C=C/CCCCCCCC([O-])=O`
- spread **7.25** (n=2)
    -   -0.50  trisodium 5-oxo-1-(4-sulfonatophenyl)-4-[(E)-2-(4-sulfonatophenyl)diazen-1-yl]-4,5-dihydro-1H-pyrazole-3-carboxylate  `[Na+].[Na+].[Na+].[O-]C(=O)C1=NN(c2ccc(cc2)[S]([O-])(=O)=O)C(=O)C1N=Nc3ccc(cc3)[S]([O-])(=O)=O`
    -   -7.75  1H-Pyrazole-3-carboxylic acid, 4,5-dihydro-5-oxo-1-(4-sulfophenyl)-4-[(4-sulfophenyl)azo]-, reaction products with guanidine hydrochloride N,N'-bis(mixed Ph, tolyl and xylyl) derivs.  `[H+].[Cl-].NC(N)=N.OC(=O)C1=NN(c2ccc(cc2)[S](O)(=O)=O)C(=O)C1N=Nc3ccc(cc3)[S](O)(=O)=O`
- spread **6.50** (n=21)
    -    0.95  dipotassium carbonate  `[K+].OC([O-])=O`
    -    0.93  dicaesium(1+) ion carbonate  `[Cs+].[Cs+].[O-]C([O-])=O`
    -    0.58  disodium carbonate  `[Na+].[Na+].[O-]C([O-])=O`
    -    0.06  sodium hydrogen carbonate  `[Na+].OC([O-])=O`
    -    0.05  trisodium hydrogen carbonate carbonate  `[Na+].[Na+].[Na+].OC([O-])=O.[O-]C([O-])=O`
    -   -0.31  tetrasodium tris(peroxol) dicarbonate  `[Na+].[Na+].[Na+].[Na+].OO.OO.OO.[O-]C([O-])=O.[O-]C([O-])=O`
    - … 외 15건
- spread **6.35** (n=4)
    -   -2.41  calcium bis(2-methyloctanoate)  `[Ca++].CCCCCCCCC([O-])=O.CCCCCCCCC([O-])=O`
    -   -2.72  nonanoic acid  `CCCCCCCCC(O)=O`
    -   -5.42  2-[(heptanoyloxy)methyl]-3-hydroxy-2-[(pentanoyloxy)methyl]propyl nonanoate  `CCCCCCCCC(O)=O.CCCCCCC(O)=O.CCCCC(O)=O.OCC(CO)(CO)CO`
    -   -8.77  Decanoic acid, ester with 2,2-bis(hydroxymethyl)-1,3-propanediol 2-ethylhexanoate octanoate  `CCCCCCCCC(O)=O.CCCCCCCC(O)=O.CCCCC(CC)C(O)=O.OCC(CO)(CO)CO`
- spread **6.30** (n=2)
    -   -0.84  disodium 2-(2,4,5,7-tetrabromo-6-oxido-3-oxo-3H-xanthen-9-yl)benzoate  `[Na+].[Na+].[O-]C(=O)c1ccccc1C2=C3C=C(Br)C(=O)C(=C3Oc4c(Br)c([O-])c(Br)cc24)Br`
    -   -7.14  dialuminium tris[2-(2,4,5,7-tetrabromo-6-oxido-3-oxoxanthen-9-yl)benzoate]  `[Al+3].[Al+3].[O-]C(=O)c1ccccc1C2=C3C=C(Br)C(=O)C(=C3Oc4c(Br)c([O-])c(Br)cc24)Br.[O-]C(=O)c5ccccc5C6=C7C=C(Br)C(=O)C(=C7Oc8c(Br)c([O-])c(Br)cc68)Br.[O-]C(=O)c9ccccc9C%10=C%11C=C(Br)C(=O)C(=C%11Oc%12c(Br)c([O-])c(Br)cc%10%12)Br`
- spread **5.92** (n=2)
    -    0.42  sodium (dimethylcarbamothioyl)sulfanide  `[Na+].CN(C)C([S-])=S`
    -   -5.50  zinc bis(dimethyldithiocarbamate)  `[Zn++].CN(C)C([S-])=S.CN(C)C([S-])=S`
- spread **5.90** (n=3)
    -    1.09  sodium cyanosulfanide  `[Na+].[S-]C#N`
    -    1.01  potassium cyanosulfanide  `[K+].[S-]C#N`
    -   -4.80  copper thiocyanate  `[Cu+].[S-]C#N`

### `B_reject`

- spread **1.72** (n=2)
    -   -4.97  5-[(4-chloro-2-nitrophenyl)diazenyl]-1-ethyl-6-hydroxy-4-methyl-2-oxo-1,2-dihydropyridine-3-carbonitrile  `CCN1C(=O)C(=N/Nc2ccc(Cl)cc2[N+]([O-])=O)/C(=C(C#N)C1=O)C`
    -   -6.69  azo dye n2;  5-(4-chloro-2-nitrophenylazo)-1-ethyl-6-hydroxy-4-methyl-2-oxo-1,2-dihydropyridine-3-carbonitrile  `CCN1C(=C(N=NC2=C(C=C(Cl)C=C2)[N+]([O-])=O)C(=C(C#N)C1=O)C)O`
- spread **1.72** (n=2)
    -   -3.06  4-hydroxy-3-(3-oxo-1-phenylbutyl)-2H-chromen-2-one  `CC(=O)CC(c1ccccc1)C2=C(O)Oc3ccccc3C2=O`
    -   -4.78  warfarin  `CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O`
- spread **1.50** (n=2)
    -    1.01  1,3-dihydroxypropan-2-one  `OCC(=O)CO`
    -   -0.49  propanal, 2,3-dihydroxy-, ()-  `OCC(O)C=O`
- spread **1.35** (n=2)
    -   -0.07  5,5-diethyl-4,6-dioxo-1H-pyrimidin-2-olate  `CCC1(C(=O)NC(=NC1=O)[O-])CC`
    -   -1.42  5,5-diethylbarbituric  acid;  veronal;  barbital;  barbitone  `CCC1(CC)C(=O)NC(=O)NC1=O`
- spread **0.61** (n=2)
    -   -3.34  p-phenylazophenol  `O=C1C=CC(C=C1)=NNc2ccccc2`
    -   -3.95  4-phenyldiazenylphenol  `OC1=CC=C(N=NC2=CC=CC=C2)C=C1`
- spread **0.57** (n=2)
    -   -0.78  1,2-benzenediol, 4-(2-propenyl)-  `Oc1ccc(CC=C)cc1O`
    -   -1.35  1,2-benzenediol, 4-(1-propenyl)-  `CC=Cc1ccc(O)c(O)c1`
- spread **0.35** (n=2)
    -   -2.93  tetracycline  `OC1=C(C(C2=C(O)[C@@](C(C(C(N)=O)=C(O)[C@H]3N(C)C)=O)(O)[C@@]3([H])C[C@]2([H])[C@@]4(O)C)=O)C4=CC=C1`
    -   -3.28  tetracycline  `CN(C)[C@H]1[C@@H]2C[C@H]3C(=C(O)c4c(O)cccc4[C@@]3(C)O)C(=O)[C@]2(O)C(=O)\C(=C(N)/O)C1=O`
- spread **0.34** (n=2)
    -   -4.16  piroxicam  `CN1C(=C(/O)Nc2ccccn2)/C(=O)c3ccccc3[S]1(=O)=O`
    -   -4.50  piroxicam  `CN1C(=C(O)C2=C(C=CC=C2)[S]1(=O)=O)C(=O)NC3=CC=CC=N3`
- spread **0.23** (n=2)
    -   -3.43  ethyl biscoumacetate  `CCOC(=O)C(C1=C(O)Oc2ccccc2C1=O)C3=C(O)Oc4ccccc4C3=O`
    -   -3.66  Ethyl_Biscoumacetate  `c1ccc2OC(=O)C(C(C(=O)OCC)C3=C(O)c4ccccc4OC3=O)=C(O)c2c1`
- spread **0.20** (n=2)
    -   -3.99  (1E)-1-(2,6,6-trimethylcyclohex-1-en-1-yl)pent-1-en-3-one; (1E)-1-(2,6,6-trimethylcyclohex-2-en-1-yl)pent-1-en-3-one; (3E)-3-methyl-4-(2,6,6-trimethylcyclohex-1-en-1-yl)but-3-en-2-one; (3E)-3-methyl-4-(2,6,6-trimethylcyclohex-2-en-1-yl)but-3-en-2-one  `CCC(=O)/C=C/C1C(=CCCC1(C)C)C`
    -   -4.19  1-(2,6,6-trimethylcyclohex-1-en-1-yl)pent-1-en-3-one  `CCC(=O)/C=C/C1=C(C)CCCC1(C)C`

### `A_strip_no_taut`

- spread **9.48** (n=11)
    -    0.78  sodium 2-ethylhexanoate  `[Na+].CCCCC(CC)C([O-])=O`
    -    0.46  zinc(2+) ion bis(2-ethylhexanoate)  `[Zn++].CCCCC(CC)C([O-])=O.CCCCC(CC)C([O-])=O`
    -   -0.91  barium(2+) ion bis(2-ethylhexanoate)  `[Ba++].CCCCC(CC)C([O-])=O.CCCCC(CC)C([O-])=O`
    -   -1.59  calcium bis(2-ethylhexanoate)  `[Ca++].CCCCC(CC)C([O-])=O.CCCCC(CC)C([O-])=O`
    -   -1.83  ({bis[(2-ethylhexanoyl)oxy]-2,4-dioxa-3-bora-1,5-dicobaltapentan-3-yl}oxy)cobaltio 2-ethylhexanoate  `CCCCC(CC)C(=O)O[Co]OB(O[Co])O[Co](OC(=O)C(CC)CCCC)OC(=O)C(CC)CCCC`
    -   -1.86  2-ethylhexanoic acid  `CCCCC(CC)C(O)=O`
    - … 외 5건
- spread **9.12** (n=2)
    -    0.76  4-hydroxybenzenesulfonic acid  `Oc1ccc(cc1)[S](O)(=O)=O`
    -   -8.36  sodium 4-hydroxybenzene-1-sulfonate  `[Na+].Oc1ccc(cc1)[S]([O-])(=O)=O`
- spread **7.62** (n=4)
    -    0.19  pentane-2,4-dione  `CC(=O)CC(C)=O`
    -   -1.27  calcium bis((2Z)-4-oxopent-2-en-2-olate)  `[Ca++].CC(=O)[CH-]C(C)=O.CC(=O)[CH-]C(C)=O`
    -   -2.07  bis(propan-2-yloxy)titaniumbis(ylium) bis((2Z)-4-oxopent-2-en-2-olate)  `[Ti+4].CC(C)[O-].CC(C)[O-].CC(=O)[CH-]C(C)=O.CC(=O)[CH-]C(C)=O`
    -   -7.44  palladium (II) di(4-oxopent-2-en-2-oate)  `[Pd++].CC(=O)[CH-]C(C)=O.CC(=O)[CH-]C(C)=O`
- spread **7.59** (n=4)
    -   -4.41  octadec-9-enoic acid - N-(2-aminoethyl)ethane-1,2-diamine (1:1)  `CCCCCCCC\C=C/CCCCCCCC(O)=O.NCCNCCN`
    -   -4.81  9-Octadecenoic acid (Z)-, reaction products with 2-[(2-aminoethyl)amino]ethanol  `CCCCCCCC\C=C/CCCCCCCC(O)=O.NCCNCCO`
    -   -6.80  lithium octadec-9-enoate  `[Li+].CCCCCCCC\C=C/CCCCCCCC([O-])=O`
    -  -12.00  copper(2+) ion bis((9Z)-octadec-9-enoate)  `[Cu++].CCCCCCCC\C=C/CCCCCCCC([O-])=O.CCCCCCCC\C=C/CCCCCCCC([O-])=O`
- spread **7.25** (n=2)
    -   -0.50  trisodium 5-oxo-1-(4-sulfonatophenyl)-4-[(E)-2-(4-sulfonatophenyl)diazen-1-yl]-4,5-dihydro-1H-pyrazole-3-carboxylate  `[Na+].[Na+].[Na+].[O-]C(=O)C1=NN(c2ccc(cc2)[S]([O-])(=O)=O)C(=O)C1N=Nc3ccc(cc3)[S]([O-])(=O)=O`
    -   -7.75  1H-Pyrazole-3-carboxylic acid, 4,5-dihydro-5-oxo-1-(4-sulfophenyl)-4-[(4-sulfophenyl)azo]-, reaction products with guanidine hydrochloride N,N'-bis(mixed Ph, tolyl and xylyl) derivs.  `[H+].[Cl-].NC(N)=N.OC(=O)C1=NN(c2ccc(cc2)[S](O)(=O)=O)C(=O)C1N=Nc3ccc(cc3)[S](O)(=O)=O`
- spread **6.50** (n=21)
    -    0.95  dipotassium carbonate  `[K+].OC([O-])=O`
    -    0.93  dicaesium(1+) ion carbonate  `[Cs+].[Cs+].[O-]C([O-])=O`
    -    0.58  disodium carbonate  `[Na+].[Na+].[O-]C([O-])=O`
    -    0.06  sodium hydrogen carbonate  `[Na+].OC([O-])=O`
    -    0.05  trisodium hydrogen carbonate carbonate  `[Na+].[Na+].[Na+].OC([O-])=O.[O-]C([O-])=O`
    -   -0.31  tetrasodium tris(peroxol) dicarbonate  `[Na+].[Na+].[Na+].[Na+].OO.OO.OO.[O-]C([O-])=O.[O-]C([O-])=O`
    - … 외 15건
- spread **6.35** (n=4)
    -   -2.41  calcium bis(2-methyloctanoate)  `[Ca++].CCCCCCCCC([O-])=O.CCCCCCCCC([O-])=O`
    -   -2.72  nonanoic acid  `CCCCCCCCC(O)=O`
    -   -5.42  2-[(heptanoyloxy)methyl]-3-hydroxy-2-[(pentanoyloxy)methyl]propyl nonanoate  `CCCCCCCCC(O)=O.CCCCCCC(O)=O.CCCCC(O)=O.OCC(CO)(CO)CO`
    -   -8.77  Decanoic acid, ester with 2,2-bis(hydroxymethyl)-1,3-propanediol 2-ethylhexanoate octanoate  `CCCCCCCCC(O)=O.CCCCCCCC(O)=O.CCCCC(CC)C(O)=O.OCC(CO)(CO)CO`
- spread **6.30** (n=2)
    -   -0.84  disodium 2-(2,4,5,7-tetrabromo-6-oxido-3-oxo-3H-xanthen-9-yl)benzoate  `[Na+].[Na+].[O-]C(=O)c1ccccc1C2=C3C=C(Br)C(=O)C(=C3Oc4c(Br)c([O-])c(Br)cc24)Br`
    -   -7.14  dialuminium tris[2-(2,4,5,7-tetrabromo-6-oxido-3-oxoxanthen-9-yl)benzoate]  `[Al+3].[Al+3].[O-]C(=O)c1ccccc1C2=C3C=C(Br)C(=O)C(=C3Oc4c(Br)c([O-])c(Br)cc24)Br.[O-]C(=O)c5ccccc5C6=C7C=C(Br)C(=O)C(=C7Oc8c(Br)c([O-])c(Br)cc68)Br.[O-]C(=O)c9ccccc9C%10=C%11C=C(Br)C(=O)C(=C%11Oc%12c(Br)c([O-])c(Br)cc%10%12)Br`
- spread **5.92** (n=2)
    -    0.42  sodium (dimethylcarbamothioyl)sulfanide  `[Na+].CN(C)C([S-])=S`
    -   -5.50  zinc bis(dimethyldithiocarbamate)  `[Zn++].CN(C)C([S-])=S.CN(C)C([S-])=S`
- spread **5.90** (n=3)
    -    1.09  sodium cyanosulfanide  `[Na+].[S-]C#N`
    -    1.01  potassium cyanosulfanide  `[K+].[S-]C#N`
    -   -4.80  copper thiocyanate  `[Cu+].[S-]C#N`

### `B_reject_no_taut`

- spread **1.35** (n=2)
    -   -0.07  5,5-diethyl-4,6-dioxo-1H-pyrimidin-2-olate  `CCC1(C(=O)NC(=NC1=O)[O-])CC`
    -   -1.42  5,5-diethylbarbituric  acid;  veronal;  barbital;  barbitone  `CCC1(CC)C(=O)NC(=O)NC1=O`
- spread **0.18** (n=2)
    -   -0.78  alanine, n-(aminocarbonyl)-  `C[C@H](NC(N)=O)C(O)=O`
    -   -0.97  l-alanine, n-(aminocarbonyl)-  `C[C@H](NC(N)=O)C([O-])=O`
- spread **0.09** (n=2)
    -   -3.07  3,4-dimethylbenzoic acid  `Cc1ccc(cc1C)C([O-])=O`
    -   -3.16  3,4-dimethylbenzoic  acid  `CC1=C(C)C=C(C=C1)C(O)=O`
- spread **0.03** (n=2)
    -   -1.69  5-chlorofuran-2-carboxylic acid  `OC(=O)C1=CC=C(Cl)O1`
    -   -1.72  2-furancarboxylic acid, 5-chloro-  `[O-]C(=O)c1oc(Cl)cc1`
- spread **0.00** (n=2)
    -   -0.47  2-dibenzofuransulfonic acid  `[O-][S](=O)(=O)c1ccc2oc3ccccc3c2c1`
    -   -0.47  dibenzofuran-2-sulfonic acid  `O=S(=O)(O)C1=CC2=C(C=C1)OC1=C2C=CC=C1`
- spread **0.00** (n=2)
    -   -2.63  3-chloro-2-nitrobenzoic  acid  `OC(=O)C1=C(C(=CC=C1)Cl)[N+]([O-])=O`
    -   -2.63  3-chloro-2-nitrobenzoic acid  `[O-]C(=O)c1cccc(Cl)c1[N+]([O-])=O`

## 7. 측정 노이즈 하한

| 항목 | 값 |
|---|---:|
| n_sd_zero | 7,746 |
| n_sd_positive | 2,236 |
| mean | 0.301 |
| median | 0.132 |
| rms | 0.516 |
| max | 3.870 |
| n_gt_0.5 | 418 |
| n_gt_1.0 | 155 |
| n_gt_2.0 | 17 |

복수 출처 분자 2,236건의 출처 간 SD 평균 **0.301**, RMS **0.516**. RMSE와 비교할 척도는 RMS가 더 맞다.
