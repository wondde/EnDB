# 노동시장 데이터 분석 프로젝트 (v2.0)

공공 데이터(KOSIS)를 활용한 **임베디드 SQL 기반** 노동시장 분석 시스템

**주요 특징**:
- ✅ 임베디드 SQL 중심 설계
- ✅ SQL 쿼리 파일 분리 관리
- ✅ 명확한 ETL → DB → 분석 구조
- ✅ 유의미한 인사이트 5개 도출

---

## 🎯 프로젝트 개요

- **데이터 출처**: KOSIS (통계청)
- **분석 기간**: 2017년 1월 ~ 2025년 9월 (105개월)
- **분석 대상**: 전국 17개 광역시도, 8개 산업
- **핵심 기술**: Python, SQLite, 임베디드 SQL
- **데이터 품질**: NULL 0%, 병합 성공률 100%

---

## 📁 주요 파일 구조

```
employment/
├── main.py                    # 실행 파일 ⭐
├── src/                       # Python 모듈
│   ├── etl.py                 # ETL (데이터 정제)
│   ├── db_loader.py           # DB 적재
│   ├── analyzer.py            # SQL 분석
│   └── ml_models.py           # AI/ML 분석
├── sql/                       # SQL 쿼리
│   ├── create_tables_sqlite.sql   # 테이블 생성 ⭐
│   ├── insights_sqlite.sql        # 분석 쿼리 ⭐
│   └── create_tables_mysql_erd.sql # ERD용 (선택)
├── data/
│   ├── raw/                   # 원본 CSV
│   └── employment.db          # SQLite DB ⭐
└── output/ml_results/         # AI/ML 결과 (PNG)
```

**자세한 설명**: [SQL파일_설명.md](SQL파일_설명.md) 참고

## 🚀 사용 방법

### 전체 파이프라인 실행 (권장)

```bash
python main.py --mode all
```

### 단계별 실행

```bash
# ETL만
python main.py --mode etl

# DB 적재만
python main.py --mode load

# 분석만 (DB에 데이터가 있어야 함)
python main.py --mode analyze
```

---

## 📊 분석 결과

### SQL 기반 인사이트 (5가지)

1. **실업률 감소 기여 산업** - 고용 창출 효과가 큰 산업 식별
2. **실업률 변동성** - 고용 불안정 지역 파악
3. **산업 다각화 지수** - 지역별 산업 구조 균형도 (HHI)
4. **고용 회복력** - 코로나 전후(2019 vs 2024) 회복 속도
5. **경제활동참가율 변화** - 노동시장 참여 트렌드

모든 분석은 **임베디드 SQL**로 수행되며, 쿼리는 `sql/insights_sqlite.sql` 파일에 정리되어 있습니다.

### AI/ML 모델 (3가지)

1. **실업률 예측** - Random Forest & Gradient Boosting (R²=0.92)
2. **지역 클러스터링** - K-Means (3개 클러스터)
3. **상관관계 분석** - 노동시장 변수 간 관계

**실행**:
```bash
python3 main.py --mode ml
```

결과는 `output/ml_results/` 폴더에 PNG 파일로 저장됩니다.
