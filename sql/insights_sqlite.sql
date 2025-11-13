-- ============================================
-- 핵심 인사이트 도출 SQL 쿼리 (SQLite 버전)
-- ============================================

-- [인사이트 1] 실업률 감소에 가장 기여한 산업 (상관관계 분석 대용)
-- 전년 대비 고용 증가가 큰 산업 vs 실업률 변화
WITH employment_growth AS (
    SELECT
        region_id,
        industry_code,
        CAST(SUBSTR(year_month, 1, 4) AS INTEGER) as year,
        AVG(employed_persons) as avg_employed
    FROM fact_employment_by_industry_monthly
    WHERE year_month >= '2020-01'
    GROUP BY region_id, industry_code, CAST(SUBSTR(year_month, 1, 4) AS INTEGER)
),
yoy_growth AS (
    SELECT
        curr.region_id,
        curr.industry_code,
        curr.year,
        curr.avg_employed - prev.avg_employed as employment_change
    FROM employment_growth curr
    LEFT JOIN employment_growth prev
        ON curr.region_id = prev.region_id
        AND curr.industry_code = prev.industry_code
        AND curr.year = prev.year + 1
    WHERE prev.avg_employed IS NOT NULL
)
SELECT
    i.industry_name,
    COUNT(*) as observations,
    ROUND(AVG(g.employment_change), 0) as avg_employment_change,
    ROUND(SUM(g.employment_change), 0) as total_employment_change
FROM yoy_growth g
JOIN dim_industry i ON g.industry_code = i.industry_code
GROUP BY i.industry_code, i.industry_name
HAVING COUNT(*) >= 10
ORDER BY total_employment_change DESC
LIMIT 5;

-- [인사이트 2] 실업률 변동성이 큰 지역 (고용 불안정성) - SQLite 버전
-- SQLite는 STDDEV가 없으므로 직접 계산
WITH monthly_stats AS (
    SELECT
        region_id,
        AVG(unemployment_rate) as avg_rate,
        -- 표준편차 직접 계산: sqrt(avg(x^2) - avg(x)^2)
        SQRT(AVG(unemployment_rate * unemployment_rate) - AVG(unemployment_rate) * AVG(unemployment_rate)) as std_rate,
        MAX(unemployment_rate) - MIN(unemployment_rate) as rate_range
    FROM fact_unemployment_monthly
    WHERE year_month >= '2020-01'
    GROUP BY region_id
)
SELECT
    r.region_name,
    ROUND(s.avg_rate, 2) as avg_unemployment_rate,
    ROUND(s.std_rate, 2) as std_dev,
    ROUND(s.rate_range, 2) as rate_range,
    CASE
        WHEN s.std_rate > 0.5 THEN '높음'
        WHEN s.std_rate > 0.3 THEN '중간'
        ELSE '안정'
    END as volatility_level
FROM monthly_stats s
JOIN dim_region r ON s.region_id = r.region_id
ORDER BY s.std_rate DESC;

-- [인사이트 3] 산업 다각화 지수 (고용이 특정 산업에 집중되지 않은 정도)
-- 높을수록 균형있는 산업 구조
WITH region_industry_share AS (
    SELECT
        e.region_id,
        e.industry_code,
        AVG(e.employed_persons) as avg_employed
    FROM fact_employment_by_industry_monthly e
    WHERE e.year_month >= '2023-01'
    GROUP BY e.region_id, e.industry_code
),
total_employment AS (
    SELECT
        region_id,
        SUM(avg_employed) as total_employed
    FROM region_industry_share
    GROUP BY region_id
),
industry_concentration AS (
    SELECT
        s.region_id,
        SUM((s.avg_employed * 1.0 / t.total_employed) * (s.avg_employed * 1.0 / t.total_employed)) as hhi
    FROM region_industry_share s
    JOIN total_employment t ON s.region_id = t.region_id
    GROUP BY s.region_id
)
SELECT
    r.region_name,
    ROUND(1 - c.hhi, 4) as diversification_index,
    CASE
        WHEN (1 - c.hhi) > 0.9 THEN '매우 다각화'
        WHEN (1 - c.hhi) > 0.85 THEN '다각화'
        WHEN (1 - c.hhi) > 0.8 THEN '보통'
        ELSE '집중'
    END as diversification_level
FROM industry_concentration c
JOIN dim_region r ON c.region_id = r.region_id
ORDER BY diversification_index DESC;

-- [인사이트 4] 고용 회복력 분석 (코로나 전후 비교)
WITH pre_covid AS (
    SELECT
        region_id,
        AVG(employed_persons) as avg_employed_2019
    FROM fact_unemployment_monthly
    WHERE year_month LIKE '2019%'
    GROUP BY region_id
),
post_covid AS (
    SELECT
        region_id,
        AVG(employed_persons) as avg_employed_2024
    FROM fact_unemployment_monthly
    WHERE year_month LIKE '2024%'
    GROUP BY region_id
)
SELECT
    r.region_name,
    ROUND(pre.avg_employed_2019, 0) as employed_2019,
    ROUND(post.avg_employed_2024, 0) as employed_2024,
    ROUND((post.avg_employed_2024 - pre.avg_employed_2019) * 100.0 / pre.avg_employed_2019, 2) as recovery_rate_pct,
    CASE
        WHEN (post.avg_employed_2024 - pre.avg_employed_2019) * 1.0 / pre.avg_employed_2019 > 0.05 THEN '강한 회복'
        WHEN (post.avg_employed_2024 - pre.avg_employed_2019) * 1.0 / pre.avg_employed_2019 > 0 THEN '회복'
        ELSE '미회복'
    END as recovery_status
FROM pre_covid pre
JOIN post_covid post ON pre.region_id = post.region_id
JOIN dim_region r ON pre.region_id = r.region_id
ORDER BY recovery_rate_pct DESC;

-- [인사이트 5] 청년 고용 추정 (인구 대비 경제활동참가율 변화)
WITH yearly_participation AS (
    SELECT
        u.region_id,
        CAST(SUBSTR(u.year_month, 1, 4) AS INTEGER) as year,
        AVG(u.labor_force * 100.0 / p.total_pop) as participation_rate
    FROM fact_unemployment_monthly u
    JOIN fact_population_yearly p
        ON u.region_id = p.region_id
        AND CAST(SUBSTR(u.year_month, 1, 4) AS INTEGER) = p.year
    GROUP BY u.region_id, CAST(SUBSTR(u.year_month, 1, 4) AS INTEGER)
)
SELECT
    r.region_name,
    ROUND(MAX(CASE WHEN y.year = 2020 THEN y.participation_rate END), 2) as rate_2020,
    ROUND(MAX(CASE WHEN y.year = 2024 THEN y.participation_rate END), 2) as rate_2024,
    ROUND(
        MAX(CASE WHEN y.year = 2024 THEN y.participation_rate END) -
        MAX(CASE WHEN y.year = 2020 THEN y.participation_rate END),
        2
    ) as rate_change
FROM yearly_participation y
JOIN dim_region r ON y.region_id = r.region_id
WHERE y.year IN (2020, 2024)
GROUP BY r.region_id, r.region_name
HAVING MAX(CASE WHEN y.year = 2020 THEN y.participation_rate END) IS NOT NULL
   AND MAX(CASE WHEN y.year = 2024 THEN y.participation_rate END) IS NOT NULL
ORDER BY rate_change DESC;
