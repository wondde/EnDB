-- 한국 시도 노동시장 분석용 MySQL 스키마
-- 사용 전 데이터베이스를 먼저 생성한 뒤 다음을 실행한다:
--   CREATE DATABASE IF NOT EXISTS employment_analysis CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
--   USE employment_analysis;

CREATE TABLE IF NOT EXISTS dim_region (
    region_id      TINYINT UNSIGNED PRIMARY KEY,
    region_name    VARCHAR(30) NOT NULL,
    unique (region_name)
);

CREATE TABLE IF NOT EXISTS dim_industry (
    industry_code  CHAR(2) PRIMARY KEY,
    industry_name  VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_unemployment_monthly (
    region_id          TINYINT UNSIGNED NOT NULL,
    year_month         CHAR(7) NOT NULL,
    unemployment_rate  DECIMAL(5,2) NOT NULL,
    unemployment_level INT,
    labor_force        INT,
    employed_persons   INT,
    PRIMARY KEY (region_id, year_month),
    CONSTRAINT fk_unemp_region FOREIGN KEY (region_id)
        REFERENCES dim_region (region_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS fact_employment_by_industry_monthly (
    region_id        TINYINT UNSIGNED NOT NULL,
    industry_code    CHAR(2) NOT NULL,
    year_month       CHAR(7) NOT NULL,
    employed_persons INT NOT NULL,
    PRIMARY KEY (region_id, industry_code, year_month),
    CONSTRAINT fk_emp_region FOREIGN KEY (region_id)
        REFERENCES dim_region (region_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT fk_emp_industry FOREIGN KEY (industry_code)
        REFERENCES dim_industry (industry_code)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS fact_population_yearly (
    region_id   TINYINT UNSIGNED NOT NULL,
    year        SMALLINT NOT NULL,
    total_pop   INT NOT NULL,
    youth_pop   INT,
    youth_rate  DECIMAL(5,2),
    PRIMARY KEY (region_id, year),
    CONSTRAINT fk_pop_region FOREIGN KEY (region_id)
        REFERENCES dim_region (region_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS bridge_population_monthly (
    region_id   TINYINT UNSIGNED NOT NULL,
    year_month  CHAR(7) NOT NULL,
    total_pop   INT NOT NULL,
    youth_rate  DECIMAL(5,2),
    PRIMARY KEY (region_id, year_month),
    CONSTRAINT fk_bridge_pop_region FOREIGN KEY (region_id)
        REFERENCES dim_region (region_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);
