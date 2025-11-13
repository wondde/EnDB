-- ============================================
-- 노동시장 데이터 분석 SQL 쿼리
-- ============================================

-- [분석 1] 산업별 평균 고용 인원 (상위 10개)
-- 임베디드 SQL로 실행하여 결과를 Python에서 처리
SELECT
    i.industry_code,
    i.industry_name,
    COUNT(*) as data_points,
    ROUND(AVG(e.employed_persons), 0) as avg_employed,
    ROUND(MIN(e.employed_persons), 0) as min_employed,
    ROUND(MAX(e.employed_persons), 0) as max_employed
FROM fact_employment_by_industry_monthly e
JOIN dim_industry i ON e.industry_code = i.industry_code
GROUP BY i.industry_code, i.industry_name
ORDER BY avg_employed DESC
LIMIT 10;

-- [분석 2] 지역별 평균 실업률 (2017-2025)
SELECT
    r.region_name,
    COUNT(*) as months,
    ROUND(AVG(u.unemployment_rate), 2) as avg_rate,
    ROUND(MIN(u.unemployment_rate), 2) as min_rate,
    ROUND(MAX(u.unemployment_rate), 2) as max_rate,
    ROUND(STDDEV(u.unemployment_rate), 2) as std_rate
FROM fact_unemployment_monthly u
JOIN dim_region r ON u.region_id = r.region_id
GROUP BY r.region_id, r.region_name
ORDER BY avg_rate ASC
LIMIT 10;

-- [분석 3] 산업별 고용 성장률 (2017 vs 2024)
WITH start_year AS (
    SELECT
        industry_code,
        AVG(employed_persons) as employed_2017
    FROM fact_employment_by_industry_monthly
    WHERE year_month LIKE '2017%'
    GROUP BY industry_code
),
end_year AS (
    SELECT
        industry_code,
        AVG(employed_persons) as employed_2024
    FROM fact_employment_by_industry_monthly
    WHERE year_month LIKE '2024%'
    GROUP BY industry_code
)
SELECT
    i.industry_code,
    i.industry_name,
    ROUND(s.employed_2017, 0) as employed_2017,
    ROUND(e.employed_2024, 0) as employed_2024,
    ROUND((e.employed_2024 - s.employed_2017) / s.employed_2017 * 100, 2) as growth_rate_pct
FROM start_year s
JOIN end_year e ON s.industry_code = e.industry_code
JOIN dim_industry i ON s.industry_code = i.industry_code
ORDER BY growth_rate_pct DESC;

-- [분석 4] 실업률이 가장 낮은 지역 Top 5 (최근 1년)
SELECT
    r.region_name,
    ROUND(AVG(u.unemployment_rate), 2) as avg_rate_2024
FROM fact_unemployment_monthly u
JOIN dim_region r ON u.region_id = r.region_id
WHERE u.year_month >= '2024-01'
GROUP BY r.region_id, r.region_name
ORDER BY avg_rate_2024 ASC
LIMIT 5;

-- [분석 5] 월별 전국 실업률 추이 (최근 24개월)
SELECT
    year_month,
    ROUND(AVG(unemployment_rate), 2) as avg_unemployment_rate,
    ROUND(AVG(labor_force), 0) as avg_labor_force
FROM fact_unemployment_monthly
WHERE year_month >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 24 MONTH), '%Y-%m')
GROUP BY year_month
ORDER BY year_month;

-- [분석 6] 산업별 고용 변동성 (표준편차)
SELECT
    i.industry_code,
    i.industry_name,
    COUNT(DISTINCT e.region_id) as regions,
    ROUND(AVG(e.employed_persons), 0) as avg_employed,
    ROUND(STDDEV(e.employed_persons), 0) as std_employed,
    ROUND(STDDEV(e.employed_persons) / AVG(e.employed_persons) * 100, 2) as cv_pct
FROM fact_employment_by_industry_monthly e
JOIN dim_industry i ON e.industry_code = i.industry_code
GROUP BY i.industry_code, i.industry_name
HAVING COUNT(*) >= 100
ORDER BY cv_pct DESC
LIMIT 10;

-- [분석 7] 인구 대비 경제활동참가율 (지역별)
SELECT
    r.region_name,
    p.year,
    ROUND(AVG(u.labor_force / p.total_pop * 100), 2) as participation_rate
FROM fact_unemployment_monthly u
JOIN dim_region r ON u.region_id = r.region_id
JOIN fact_population_yearly p
    ON u.region_id = p.region_id
    AND CAST(SUBSTRING(u.year_month, 1, 4) AS UNSIGNED) = p.year
GROUP BY r.region_id, r.region_name, p.year
HAVING p.year >= 2020
ORDER BY r.region_name, p.year;
