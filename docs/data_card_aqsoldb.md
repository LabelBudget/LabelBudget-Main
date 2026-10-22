# 데이터 카드 — aqsoldb

- 과제: regression · 라벨: `Solubility` (log10(mol/L))
- 생성: 2026-10-22T05:37:41+00:00

## 1. 단계별 생존 개수

| 단계 | 개수 | 직전 대비 |
|---|---:|---:|
| 원본 | 9,982 |  |
| 결측 제거 | 9,982 | +0 |
| 표준화 | 8,735 | -1,247 |
| 중복 제거 | 8,703 | -32 |

최종 보존율 **87.2%**

## 2. 표준화 거부 사유

| 사유 | 개수 |
|---|---:|
| `multi_component` | 1,141 |
| `non_organic_element` | 53 |
| `no_carbon` | 25 |
| `too_small` | 14 |
| `molwt_exceeded` | 11 |
| `parse_failed` | 2 |
| `too_large` | 1 |

## 3. 구조 변경 내역

SMILES가 바뀐 분자: **582**

| 변경 유형 | 개수 |
|---|---:|
| `tautomer_changed` | 488 |
| `uncharged` | 65 |
| `cleanup_changed` | 32 |

## 4. 중복 및 라벨 충돌

| 항목 | 값 |
|---|---:|
| 완전 중복 행 | 26 |
| 고유 분자 | 8,709 |
| 충돌 분자 | 6 |
| 충돌로 제거 | 6 |
| 규칙 / tolerance | `drop` / 0.5 |
| 골격 충돌 그룹 / 분자 | 258 / 549 |

> ⚠️ 골격(InChIKey 첫 블록)이 같은 분자는 제거하지 않았다. train/test로 갈라지면 누수가 되므로 분할 후 `leakage_check`로 점검한다.

충돌 상위 예시:

| key | n | 라벨 | spread | 제거 |
|---|---:|---|---:|---|
| `FHERFUXQYBSNRD-UHFFFAOYSA-N` | 2 | -4.967, -6.690 | 1.723 | 예 |
| `PJVWKTKQMONHTI-UHFFFAOYSA-N` | 2 | -3.063, -4.780 | 1.717 | 예 |
| `RXKJFZQQPQGTFL-UHFFFAOYSA-N` | 2 | 1.014, -0.489 | 1.503 | 예 |
| `FTOAOBMCPZCFFF-UHFFFAOYSA-N` | 2 | -1.420, -0.070 | 1.350 | 예 |
| `BEYOBVMPDRKTNR-UHFFFAOYSA-N` | 2 | -3.343, -3.950 | 0.607 | 예 |
| `MDXYNRAMDATLBT-UHFFFAOYSA-N` | 2 | -0.779, -1.351 | 0.572 | 예 |

## 5. 표준화가 만들어낸 중복

| 발생 단계 | 그룹 수 |
|---|---:|
| 들어온 중복 (입력 InChIKey 동일) | 0 |
| 표준화가 만든 중복 | 26 |
| └ 그중 토토머 정규화가 관여 | 20 |

> 표준 InChI는 이동 수소 토토머(예: 2-하이드록시피리딘/2-피리돈)를 이미 같은 키로 보므로 그런 쌍은 '들어온 중복'에 집계된다. '토토머 정규화가 관여'는 RDKit 규칙으로만 합쳐진 것(예: 케토-엔올)이다.

### 표준화가 만든 중복의 라벨 불일치

| 항목 | 값 |
|---|---:|
| n_groups | 26 |
| mean | 0.356 |
| median | 0.093 |
| max | 1.723 |
| n_gt_0.5 | 6 |
| n_gt_1.0 | 4 |
| n_gt_2.0 | 0 |

### 토토머 정규화가 관여한 중복

| 항목 | 값 |
|---|---:|
| n_groups | 20 |
| mean | 0.379 |
| median | 0.135 |
| max | 1.723 |
| n_gt_0.5 | 5 |
| n_gt_1.0 | 3 |
| n_gt_2.0 | 0 |

> 토토머 규칙은 이중결합 위치 이성질체처럼 **서로 다른 화합물**을 합칠 수 있다. 아래 예시를 확인하고, spread가 tolerance 이하라 병합된 그룹은 한계로 기록한다.

| spread | 라벨 | 입력 SMILES |
|---:|---|---|
| 1.723 | -4.967, -6.690 | `CCN1C(=O)C(=N/Nc2ccc(Cl)cc2[N+]([O-])=O)/C(=C(C#N)C1=O)C`<br>`CCN1C(=C(N=NC2=C(C=C(Cl)C=C2)[N+]([O-])=O)C(=C(C#N)C1=O)C)O` |
| 1.717 | -3.063, -4.780 | `CC(=O)CC(c1ccccc1)C2=C(O)Oc3ccccc3C2=O`<br>`CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O` |
| 1.503 | 1.014, -0.489 | `OCC(=O)CO`<br>`OCC(O)C=O` |
| 0.607 | -3.343, -3.950 | `O=C1C=CC(C=C1)=NNc2ccccc2`<br>`OC1=CC=C(N=NC2=CC=CC=C2)C=C1` |
| 0.572 | -0.779, -1.351 | `Oc1ccc(CC=C)cc1O`<br>`CC=Cc1ccc(O)c(O)c1` |
| 0.354 | -3.284, -2.930 | `CN(C)[C@H]1[C@@H]2C[C@H]3C(=C(O)c4c(O)cccc4[C@@]3(C)O)C(=O)[C@]2(O)C(=O)\C(=C(N)/O)C1=O`<br>`OC1=C(C(C2=C(O)[C@@](C(C(C(N)=O)=C(O)[C@H]3N(C)C)=O)(O)[C@@]3([H])C[C@]2([H])[C@@]4(O)C)=O)C4=CC=C1` |
| 0.341 | -4.159, -4.500 | `CN1C(=C(/O)Nc2ccccn2)/C(=O)c3ccccc3[S]1(=O)=O`<br>`CN1C(=C(O)C2=C(C=CC=C2)[S]1(=O)=O)C(=O)NC3=CC=CC=N3` |
| 0.234 | -3.426, -3.660 | `CCOC(=O)C(C1=C(O)Oc2ccccc2C1=O)C3=C(O)Oc4ccccc4C3=O`<br>`c1ccc2OC(=O)C(C(C(=O)OCC)C3=C(O)c4ccccc4OC3=O)=C(O)c2c1` |
| 0.195 | -4.187, -3.992 | `CCC(=O)/C=C/C1=C(C)CCCC1(C)C`<br>`CCC(=O)/C=C/C1C(=CCCC1(C)C)C` |
| 0.179 | -3.080, -3.259 | `CC(=O)/C=C/C1=C(C)CCCC1(C)C`<br>`CC(=O)/C=C/C1C(=CCCC1(C)C)C` |
| 0.092 | -3.630, -3.538 | `CCCC1=C(O)N2N(C1=O)C3=CC(=CC=C3N=C2N(C)C)C`<br>`CCCC1C(=O)N3N(C1=O)c2cc(C)ccc2N=C3N(C)C ` |
| 0.050 | -3.030, -3.080 | `OC/1=CC(=O)C=CC1=N/Nc2ccccc2`<br>`OC1=CC(O)=C(N=NC2=CC=CC=C2)C=C1` |
| 0.006 | -3.056, -3.050 | `OC(=O)C1=C/C(C=CC1=O)=N/Nc2ccccc2`<br>`O=C(O)C1=C(O)C=CC(N=NC2=CC=CC=C2)=C1` |
| 0.004 | -4.864, -4.860 | `OC1=C(C2CCCc3ccccc23)C(=O)c4ccccc4O1`<br>`OC1=C(C2CCCC3=CC=CC=C23)C(=O)OC4=CC=CC=C14` |
| 0.003 | 0.280, 0.277 | `OCC(O)C1OC(=O)C(=C1O)O`<br>`OCC(O)C1OC(=C(O)C1=O)O` |
| 0.003 | -5.836, -5.839 | `CC(=O)CC(c1ccc(Cl)cc1)C2=C(O)Oc3ccccc3C2=O`<br>`CC(=O)CC(c1ccc(Cl)cc1)c2c(O)c3ccccc3oc2=O` |
| 0.003 | -2.937, -2.940 | `Oc1ccc2ccccc2c1N=O`<br>`O=C1C=CC2=C(C=CC=C2)C1=NO` |
| 0.001 | -2.511, -2.510 | `OC1=CC(=O)C[C@@H](C1)c2ccccc2`<br>`O=C1CC(=O)CC(C2=CC=CC=C2)C1` |
| 0.000 | 0.586, 0.586 | `O=C1NC=NC=C1`<br>`O=C1CC=NC=N1` |
| 0.000 | -2.500, -2.500 | `CNC(/NCCSCC1=CC=C(CN(C)C)O1)=C/[N+]([O-])=O`<br>`S(Cc1oc(CN(C)C)cc1)CCN\C(NC)=C\N(=O)=O` |

## 6. 라벨 분포

| 통계 | 값 |
|---|---:|
| n | 8703 |
| mean | -3.007 |
| std | 2.305 |
| min | -13.172 |
| p25 | -4.359 |
| median | -2.706 |
| p75 | -1.420 |
| max | 2.138 |

## 7. 표준화 설정

```json
{
  "stereo": "keep",
  "canonicalize_tautomer": true,
  "strip_salts": false,
  "reject_multi_component": true,
  "organic_only": true,
  "min_heavy_atoms": 3,
  "max_heavy_atoms": 100,
  "max_mol_weight": 1000.0,
  "max_tautomers": 200,
  "max_transforms": 200
}
```

순서 고정: 파싱 → Cleanup → [염 제거 | 다단편 거부] → Uncharger → 토토머 → 입체 → 필터. 순서가 바뀌면 결과가 바뀐다.
