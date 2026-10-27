# 데이터 카드: ZINC250k 사전학습 세트

- 출처: `https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv`
- SHA-256: `35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae`
- 용도: 자기지도 사전학습 전용 (라벨 없음). 스펙 05 §2 P1·P2
- 생성: `python scripts/curate_pretrain.py`

| 단계 | 분자 수 |
|---|---:|
| 원본 | 249,455 |
| 표준화 통과 (평가 데이터와 같은 설정) | 249,398 |
| InChIKey 중복 제거 | 249,334 |
| 평가 데이터와 골격 일치 제거 | 249,219 |

표준화 거부 사유: inchikey_failed 57

평가 데이터셋별 골격(InChIKey 첫 블록) 일치 (한 분자가 여러 세트와 겹칠 수 있다): aqsoldb 75, tox21_SR-ARE 73, esol 16 → 합집합 115개 제거

평가 분할은 시드마다 바뀌어 'test 분자만' 뺄 수 없으므로, 평가 데이터 전체와 겹치는 골격을 뺐다.
