-- 한국 시도 노동시장 분석용 SQLite 스키마
-- SQLite는 별도 데이터베이스 생성 불필요 (파일 기반)

CREATE TABLE IF NOT EXISTS dim_region (
    region_id      INTEGER PRIMARY KEY,
    region_name    TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_industry (
    industry_code  TEXT PRIMARY KEY,
    industry_name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_unemployment_monthly (
    region_id          INTEGER NOT NULL,
    year_month         TEXT NOT NULL,
    unemployment_rate  REAL NOT NULL,
    unemployment_level INTEGER,
    labor_force        INTEGER,
    employed_persons   INTEGER,
    PRIMARY KEY (region_id, year_month),
    FOREIGN KEY (region_id) REFERENCES dim_region (region_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS fact_employment_by_industry_monthly (
    region_id        INTEGER NOT NULL,
    industry_code    TEXT NOT NULL,
    year_month       TEXT NOT NULL,
    employed_persons INTEGER NOT NULL,
    PRIMARY KEY (region_id, industry_code, year_month),
    FOREIGN KEY (region_id) REFERENCES dim_region (region_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (industry_code) REFERENCES dim_industry (industry_code)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

-- 월별 인구 팩트 테이블 (신규!)
CREATE TABLE IF NOT EXISTS fact_population_monthly (
    region_id   INTEGER NOT NULL,
    year_month  TEXT NOT NULL,
    total_pop   INTEGER NOT NULL,
    PRIMARY KEY (region_id, year_month),
    FOREIGN KEY (region_id) REFERENCES dim_region (region_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

-- 연도별 인구 팩트 테이블 (기존 - 호환성 유지)
CREATE TABLE IF NOT EXISTS fact_population_yearly (
    region_id   INTEGER NOT NULL,
    year        INTEGER NOT NULL,
    total_pop   INTEGER NOT NULL,
    youth_pop   INTEGER,
    youth_rate  REAL,
    PRIMARY KEY (region_id, year),
    FOREIGN KEY (region_id) REFERENCES dim_region (region_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);
