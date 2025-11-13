-- ============================================
-- 한국 시도 노동시장 분석용 MySQL 스키마
-- MySQL Workbench ERD 생성용
-- ============================================

-- 1. 데이터베이스 생성
CREATE DATABASE IF NOT EXISTS employment_analysis
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE employment_analysis;

-- 2. 차원 테이블 (Dimension Tables)

-- 지역 차원 테이블
CREATE TABLE IF NOT EXISTS dim_region (
    region_id      TINYINT UNSIGNED PRIMARY KEY COMMENT '지역 ID',
    region_name    VARCHAR(30) NOT NULL UNIQUE COMMENT '지역명 (예: 서울특별시)'
) COMMENT='지역 차원 테이블';

-- 산업 차원 테이블 (한글 산업명 지원)
CREATE TABLE IF NOT EXISTS dim_industry (
    industry_code  VARCHAR(50) PRIMARY KEY COMMENT '산업 코드 (알파벳 또는 한글)',
    industry_name  VARCHAR(100) NOT NULL COMMENT '산업명 (예: 제조업, 사회간접자본 및 기타서비스업)'
) COMMENT='산업 차원 테이블';

-- 3. 팩트 테이블 (Fact Tables)

-- 월별 실업률 팩트 테이블
CREATE TABLE IF NOT EXISTS fact_unemployment_monthly (
    region_id          TINYINT UNSIGNED NOT NULL COMMENT '지역 ID',
    `year_month`       CHAR(7) NOT NULL COMMENT '년월 (YYYY-MM)',
    unemployment_rate  DECIMAL(5,2) NOT NULL COMMENT '실업률 (%)',
    unemployment_level INT COMMENT '실업자 수 (명)',
    labor_force        INT COMMENT '경제활동인구 (명)',
    employed_persons   INT COMMENT '취업자 수 (명)',
    PRIMARY KEY (region_id, `year_month`),
    CONSTRAINT fk_unemp_region FOREIGN KEY (region_id)
        REFERENCES dim_region (region_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
) COMMENT='월별 지역별 실업률 데이터';

-- 월별 산업별 고용 팩트 테이블
CREATE TABLE IF NOT EXISTS fact_employment_by_industry_monthly (
    region_id        TINYINT UNSIGNED NOT NULL COMMENT '지역 ID',
    industry_code    VARCHAR(50) NOT NULL COMMENT '산업 코드',
    `year_month`     CHAR(7) NOT NULL COMMENT '년월 (YYYY-MM)',
    employed_persons INT NOT NULL COMMENT '산업별 취업자 수 (명)',
    PRIMARY KEY (region_id, industry_code, `year_month`),
    CONSTRAINT fk_emp_region FOREIGN KEY (region_id)
        REFERENCES dim_region (region_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_emp_industry FOREIGN KEY (industry_code)
        REFERENCES dim_industry (industry_code)
        ON UPDATE CASCADE ON DELETE RESTRICT
) COMMENT='월별 지역별 산업별 고용 데이터';

-- 월별 인구 팩트 테이블 (신규!)
CREATE TABLE IF NOT EXISTS fact_population_monthly (
    region_id   TINYINT UNSIGNED NOT NULL COMMENT '지역 ID',
    `year_month` CHAR(7) NOT NULL COMMENT '년월 (YYYY-MM)',
    total_pop   INT NOT NULL COMMENT '총 인구 (명)',
    PRIMARY KEY (region_id, `year_month`),
    CONSTRAINT fk_pop_monthly_region FOREIGN KEY (region_id)
        REFERENCES dim_region (region_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
) COMMENT='월별 지역별 인구 데이터 (실제 월별 데이터)';

-- 연도별 인구 팩트 테이블 (기존 - 호환성 유지)
CREATE TABLE IF NOT EXISTS fact_population_yearly (
    region_id   TINYINT UNSIGNED NOT NULL COMMENT '지역 ID',
    year        SMALLINT NOT NULL COMMENT '연도',
    total_pop   INT NOT NULL COMMENT '총 인구 (명)',
    youth_pop   INT COMMENT '청년 인구 (명)',
    youth_rate  DECIMAL(5,2) COMMENT '청년 인구 비율 (%)',
    PRIMARY KEY (region_id, year),
    CONSTRAINT fk_pop_yearly_region FOREIGN KEY (region_id)
        REFERENCES dim_region (region_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
) COMMENT='연도별 지역별 인구 데이터 (월별 평균)';

-- 5. 인덱스 (성능 최적화)
-- IF NOT EXISTS 사용으로 중복 생성 방지
CREATE INDEX IF NOT EXISTS idx_unemployment_date ON fact_unemployment_monthly(`year_month`);
CREATE INDEX IF NOT EXISTS idx_employment_date ON fact_employment_by_industry_monthly(`year_month`);
CREATE INDEX IF NOT EXISTS idx_employment_industry ON fact_employment_by_industry_monthly(industry_code);
