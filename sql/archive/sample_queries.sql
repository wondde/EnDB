-- 탐색적 분석을 위한 SQL 조인 예시

-- 1. 실업률과 인구 브리지 테이블을 조인해 인구·실업률을 동시에 조회
SELECT
    u.region_id,
    u.year_month,
    u.unemployment_rate,
    u.labor_force,
    b.total_pop,
    b.youth_rate
FROM fact_unemployment_monthly AS u
JOIN bridge_population_monthly AS b
  ON u.region_id = b.region_id
 AND u.year_month = b.year_month
WHERE u.year_month BETWEEN '2021-01' AND '2024-12';

-- 2. 산업별 취업자 추이와 실업률을 동시에 확인
SELECT
    u.region_id,
    r.region_name,
    u.year_month,
    i.industry_name,
    e.employed_persons,
    u.unemployment_rate
FROM fact_unemployment_monthly AS u
JOIN fact_employment_by_industry_monthly AS e
  ON u.region_id = e.region_id
 AND u.year_month = e.year_month
JOIN dim_industry AS i
  ON e.industry_code = i.industry_code
JOIN dim_region AS r
  ON u.region_id = r.region_id
WHERE i.industry_code = 'C';

-- 3. 청년 비율과 실업률의 상관관계를 지역별로 요약
SELECT
    r.region_name,
    AVG(u.unemployment_rate) AS avg_unemployment_rate,
    AVG(b.youth_rate) AS avg_youth_rate,
    CORR(u.unemployment_rate, b.youth_rate) AS corr_rate_youth
FROM fact_unemployment_monthly AS u
JOIN bridge_population_monthly AS b
  ON u.region_id = b.region_id
 AND u.year_month = b.year_month
JOIN dim_region AS r
  ON u.region_id = r.region_id
GROUP BY r.region_name
ORDER BY avg_unemployment_rate DESC;

-- 4. 지역·산업별 취업자 수 전년 대비 증감률 계산
SELECT
    e.region_id,
    r.region_name,
    e.industry_code,
    i.industry_name,
    e.year_month,
    e.employed_persons,
    e.employed_persons
      / LAG(e.employed_persons, 12) OVER (PARTITION BY e.region_id, e.industry_code ORDER BY e.year_month)
      - 1 AS employment_growth_yoy
FROM fact_employment_by_industry_monthly AS e
JOIN dim_region AS r
  ON e.region_id = r.region_id
JOIN dim_industry AS i
  ON e.industry_code = i.industry_code
WHERE e.year_month >= '2018-01';

-- 5. 이동평균 대비 실업률이 급등한 달을 탐지
WITH rolling AS (
    SELECT
        u.region_id,
        u.year_month,
        u.unemployment_rate,
        AVG(u.unemployment_rate) OVER (PARTITION BY u.region_id ORDER BY u.year_month ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)
            AS ma_6m
    FROM fact_unemployment_monthly AS u
)
SELECT
    r.region_name,
    year_month,
    unemployment_rate,
    ma_6m,
    unemployment_rate - ma_6m AS deviation
FROM rolling
JOIN dim_region AS r
  ON rolling.region_id = r.region_id
WHERE rolling.unemployment_rate - rolling.ma_6m > 1.0
ORDER BY deviation DESC;
