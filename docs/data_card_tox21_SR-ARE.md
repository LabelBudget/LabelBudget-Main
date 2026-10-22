# 데이터 카드 — tox21

- 과제: classification · 라벨: `SR-ARE` (binary)
- 생성: 2026-10-22T06:34:47+00:00

## 1. 단계별 생존 개수

| 단계 | 개수 | 직전 대비 |
|---|---:|---:|
| 원본 | 7,831 |  |
| 결측 제거 | 5,832 | -1,999 |
| 표준화 | 5,576 | -256 |
| 중복 제거 | 5,514 | -62 |

최종 보존율 **70.4%**

## 2. 표준화 거부 사유

| 사유 | 개수 |
|---|---:|
| `multi_component` | 171 |
| `non_organic_element` | 40 |
| `no_carbon` | 21 |
| `molwt_exceeded` | 13 |
| `parse_failed` | 7 |
| `too_small` | 2 |
| `too_large` | 1 |
| `inchikey_failed` | 1 |

## 3. 구조 변경 내역

SMILES가 바뀐 분자: **489**

| 변경 유형 | 개수 |
|---|---:|
| `tautomer_changed` | 275 |
| `uncharged` | 218 |
| `cleanup_changed` | 25 |

## 4. 중복 및 라벨 충돌

| 항목 | 값 |
|---|---:|
| 완전 중복 행 | 59 |
| 고유 분자 | 5,517 |
| 충돌 분자 | 3 |
| 충돌로 제거 | 3 |
| 규칙 / tolerance | `majority` / — |
| 골격 충돌 그룹 / 분자 | 62 / 132 |

> ⚠️ 골격(InChIKey 첫 블록)이 같은 분자는 제거하지 않았다. train/test로 갈라지면 누수가 되므로 분할 후 `leakage_check`로 점검한다.

충돌 상위 예시:

| key | n | 라벨 | spread | 제거 |
|---|---:|---|---:|---|
| `LLEMOWNGBBNAJR-UHFFFAOYSA-N` | 2 | 0.000, 1.000 | 1.000 | 예 |
| `LCTONWCANYUPML-UHFFFAOYSA-N` | 2 | 0.000, 1.000 | 1.000 | 예 |
| `BACYUWVYYTXETD-UHFFFAOYSA-N` | 2 | 0.000, 1.000 | 1.000 | 예 |

## 5. 표준화가 만들어낸 중복

| 발생 단계 | 그룹 수 |
|---|---:|
| 들어온 중복 (입력 InChIKey 동일) | 2 |
| 표준화가 만든 중복 | 57 |
| └ 그중 토토머 정규화가 관여 | 4 |

> 표준 InChI는 이동 수소 토토머(예: 2-하이드록시피리딘/2-피리돈)를 이미 같은 키로 보므로 그런 쌍은 '들어온 중복'에 집계된다. '토토머 정규화가 관여'는 RDKit 규칙으로만 합쳐진 것(예: 케토-엔올)이다.

### 표준화가 만든 중복의 라벨 불일치

| 항목 | 값 |
|---|---:|
| n_groups | 57 |
| n_label_disagree | 3 |

### 토토머 정규화가 관여한 중복

| 항목 | 값 |
|---|---:|
| n_groups | 4 |
| n_label_disagree | 0 |

> 토토머 규칙은 이중결합 위치 이성질체처럼 **서로 다른 화합물**을 합칠 수 있다. 아래 예시를 확인하고, spread가 tolerance 이하라 병합된 그룹은 한계로 기록한다.

| spread | 라벨 | 입력 SMILES |
|---:|---|---|
| 0.000 | 0.000, 0.000 | `CC(=O)/C=C/C1=C(C)CCCC1(C)C`<br>`CC(=O)/C=C/C1C(C)=CCCC1(C)C` |
| 0.000 | 0.000, 0.000 | `CC(=O)C1C(=O)C=C(C)OC1=O`<br>`CC1=CC(=O)/C(=C(/C)[O-])C(=O)O1` |
| 0.000 | 0.000, 0.000 | `O=C1C2:C(O):C:C:C:C:2CC2:C:C:C:C(O):C:21`<br>`OC1:C:C:C:C2:C:C3:C:C:C:C(O):C:3:C(O):C:1:2` |
| 0.000 | 0.000, 0.000 | `O=C1NC(=O)C2CC=CCC12`<br>`O=C1NC(=O)[C@@H]2CC=CC[C@H]12` |

## 6. 라벨 분포

| 통계 | 값 |
|---|---:|
| n | 5514 |
| n_positive | 872 |
| n_negative | 4642 |
| positive_rate | 0.158 |
| imbalance_ratio | 5.323 |

> ⚠️ 음성:양성 = 5.3:1 로 불균형이 심하므로 AUROC 단독 보고는 낙관적이다. AUPRC와 PPV/NPV를 병기한다.

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
