# 데이터 카드 — esol

- 과제: regression · 라벨: `measured log solubility in mols per litre` (log10(mol/L))
- 생성: 2026-10-22T05:38:27+00:00

## 1. 단계별 생존 개수

| 단계 | 개수 | 직전 대비 |
|---|---:|---:|
| 원본 | 1,128 |  |
| 결측 제거 | 1,128 | +0 |
| 표준화 | 1,121 | -7 |
| 중복 제거 | 1,110 | -11 |

최종 보존율 **98.4%**

## 2. 표준화 거부 사유

| 사유 | 개수 |
|---|---:|
| `too_small` | 7 |

## 3. 구조 변경 내역

SMILES가 바뀐 분자: **67**

| 변경 유형 | 개수 |
|---|---:|
| `tautomer_changed` | 63 |
| `cleanup_changed` | 4 |

## 4. 중복 및 라벨 충돌

| 항목 | 값 |
|---|---:|
| 완전 중복 행 | 11 |
| 고유 분자 | 1,110 |
| 충돌 분자 | 1 |
| 충돌로 제거 | 0 |
| 규칙 / tolerance | `median` / 0.5 |
| 골격 충돌 그룹 / 분자 | 2 / 4 |

> ⚠️ 골격(InChIKey 첫 블록)이 같은 분자는 제거하지 않았다. train/test로 갈라지면 누수가 되므로 분할 후 `leakage_check`로 점검한다.

충돌 상위 예시:

| key | n | 라벨 | spread | 제거 |
|---|---:|---|---:|---|
| `FBPFZTCFMRRESA-UHFFFAOYSA-N` | 2 | 0.060, 1.090 | 1.030 | 아니오 |

## 5. 표준화가 만들어낸 중복

| 발생 단계 | 그룹 수 |
|---|---:|
| 들어온 중복 (입력 InChIKey 동일) | 11 |
| 표준화가 만든 중복 | 0 |
| └ 그중 토토머 정규화가 관여 | 0 |

> 표준 InChI는 이동 수소 토토머(예: 2-하이드록시피리딘/2-피리돈)를 이미 같은 키로 보므로 그런 쌍은 '들어온 중복'에 집계된다. '토토머 정규화가 관여'는 RDKit 규칙으로만 합쳐진 것(예: 케토-엔올)이다.

### 표준화가 만든 중복의 라벨 불일치

없음.

### 토토머 정규화가 관여한 중복

없음.

## 6. 라벨 분포

| 통계 | 값 |
|---|---:|
| n | 1110 |
| mean | -3.067 |
| std | 2.090 |
| min | -11.600 |
| p25 | -4.325 |
| median | -2.895 |
| p75 | -1.616 |
| max | 1.580 |

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
