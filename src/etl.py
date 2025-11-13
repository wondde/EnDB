"""
ETL (Extract, Transform, Load) 모듈

역할: 원본 CSV 데이터를 정제하여 DataFrame으로 반환
"""

from pathlib import Path
from typing import Dict, Tuple
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# 지역 코드 매핑
REGION_CODE_MAP: Dict[str, int] = {
    "서울특별시": 11, "부산광역시": 26, "대구광역시": 27, "인천광역시": 28,
    "광주광역시": 29, "대전광역시": 30, "울산광역시": 31, "세종특별자치시": 36,
    "경기도": 41, "강원도": 42, "충청북도": 43, "충청남도": 44,
    "전라북도": 45, "전라남도": 46, "경상북도": 47, "경상남도": 48,
    "제주특별자치도": 50,
}


def extract_unemployment() -> pd.DataFrame:
    """실업률 데이터 추출 및 정제"""

    csv_path = DATA_DIR / "unemployment.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"시점": str})
    logger.info(f"✓ 실업률 원본 로드: {df.shape}")

    # Wide → Long 변환
    id_vars = ["시점", "항목"]
    region_cols = [c for c in df.columns if c not in id_vars]
    tidy = df.melt(id_vars=id_vars, value_vars=region_cols, var_name="region_name", value_name="value")

    # 날짜 형식 통일 (2017.1 → 2017-01)
    date_parts = tidy["시점"].astype(str).str.strip().str.split(".", expand=True)
    tidy["year_month"] = date_parts[0] + "-" + date_parts[1].str.zfill(2)

    # 지역명 정규화
    tidy["region_name"] = tidy["region_name"].str.strip().replace({"제주도": "제주특별자치도"})
    tidy["metric"] = tidy["항목"].str.strip()
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")

    # 단위 통일 (천명 → 명)
    thousand_metrics = {"경제활동인구 (천명)", "취업자 (천명)", "실업자 (천명)", "15세이상인구 (천명)"}
    tidy.loc[tidy["metric"].isin(thousand_metrics), "value"] *= 1_000

    # Pivot
    pivot = tidy.pivot_table(
        index=["region_name", "year_month"],
        columns="metric",
        values="value",
        aggfunc="first"
    ).reset_index()

    # 컬럼명 정리
    pivot = pivot.rename(columns={
        "실업률 (%)": "unemployment_rate",
        "실업자 (천명)": "unemployment_level",
        "경제활동인구 (천명)": "labor_force",
        "취업자 (천명)": "employed_persons",
    })

    # 지역 코드 매핑
    pivot["region_id"] = pivot["region_name"].map(REGION_CODE_MAP)
    pivot = pivot.dropna(subset=["region_id"])
    pivot["region_id"] = pivot["region_id"].astype(int)

    logger.info(f"✓ 실업률 데이터 정제 완료: {len(pivot)}행, {pivot['region_id'].nunique()}개 지역")

    return pivot[["region_id", "year_month", "unemployment_rate", "unemployment_level", "labor_force", "employed_persons"]]


def extract_employment() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """산업별 고용 데이터 추출 및 정제"""

    csv_path = DATA_DIR / "employment_industry.csv"
    df = pd.read_csv(csv_path, encoding="cp949")
    logger.info(f"✓ 산업별 고용 원본 로드: {df.shape}")

    # 컬럼명 정리
    df.columns = [c.strip().replace(" 월", "") for c in df.columns]
    value_cols = [c for c in df.columns if c not in {"시도별", "산업별", "항목", "단위"}]

    # 취업자 데이터만 필터링
    df = df[df["항목"].astype(str).str.contains("취업자") & df["단위"].astype(str).str.contains("천명")]

    # Wide → Long 변환
    tidy = df.melt(
        id_vars=["시도별", "산업별"],
        value_vars=value_cols,
        var_name="year_month",
        value_name="employed"
    )

    tidy["year_month"] = tidy["year_month"].str.replace(".", "-", regex=False)
    tidy["region_name"] = tidy["시도별"].astype(str).str.strip().replace({"제주도": "제주특별자치도"})
    tidy["region_id"] = tidy["region_name"].map(REGION_CODE_MAP)
    tidy["employed"] = pd.to_numeric(tidy["employed"], errors="coerce") * 1_000

    # 산업 코드 추출 (단일 + 복합)
    tidy["industry_raw"] = tidy["산업별"].astype(str).str.strip()
    industry_info = tidy["industry_raw"].str.extract(r"^(?P<industry_code>[A-Z])\s+(?P<industry_name>.+)$")

    # 복합 그룹 처리
    mask_compound = industry_info["industry_code"].isna() & tidy["industry_raw"].str.contains(r"^\*", na=False)
    compound_info = tidy.loc[mask_compound, "industry_raw"].str.extract(r"^\*\s*(?P<industry_name>[^(]+)")
    industry_info.loc[mask_compound, "industry_name"] = compound_info["industry_name"].str.strip()
    industry_info.loc[mask_compound, "industry_code"] = compound_info["industry_name"].str.strip()

    tidy = pd.concat([tidy, industry_info], axis=1)

    # Fact 테이블 (employed_persons가 NULL인 행 제거)
    fact = tidy.dropna(subset=["region_id", "industry_code", "year_month", "employed"])
    fact = fact[["region_id", "industry_code", "year_month", "employed"]].rename(columns={"employed": "employed_persons"})
    fact["region_id"] = fact["region_id"].astype(int)

    # Dimension 테이블
    dim_industry = (
        tidy[["industry_code", "industry_name"]]
        .dropna(subset=["industry_code"])
        .drop_duplicates()
        .sort_values("industry_code")
    )

    logger.info(f"✓ 산업별 고용 데이터 정제 완료: {len(fact)}행, {dim_industry['industry_code'].nunique()}개 산업")

    return fact, dim_industry


def extract_population() -> tuple[pd.DataFrame, pd.DataFrame]:
    """인구 데이터 추출 및 정제 - 월별 + 연도별 반환"""

    csv_path = DATA_DIR / "population.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig", header=[0, 1])
    logger.info(f"✓ 인구 원본 로드: {df.shape}")

    # 컬럼명 정리
    df.columns = pd.MultiIndex.from_tuples(
        [(str(c[0]).strip(), str(c[1]).strip()) for c in df.columns],
        names=["year_month_raw", "metric"]
    )

    region_series = df[("행정구역(시군구)별", "행정구역(시군구)별")].astype(str).str.strip()
    mask = region_series != "전국"
    region_series = region_series[mask]
    df = df.loc[mask].drop(columns=[("행정구역(시군구)별", "행정구역(시군구)별")])

    # 월별 데이터로 변환
    monthly = []
    for month_raw, metric in df.columns:
        month_clean = month_raw.replace(".", "-")
        temp = pd.DataFrame({
            "region_name": region_series,
            "year_month": month_clean,
            "metric": metric,
            "value": df[(month_raw, metric)]
        })
        monthly.append(temp)

    monthly_df = pd.concat(monthly, ignore_index=True)
    monthly_df = monthly_df[monthly_df["metric"] == "총인구수 (명)"]
    monthly_df["value"] = pd.to_numeric(monthly_df["value"], errors="coerce")
    monthly_df["region_id"] = monthly_df["region_name"].map(REGION_CODE_MAP)
    monthly_df = monthly_df.dropna(subset=["region_id", "value"])
    monthly_df["region_id"] = monthly_df["region_id"].astype(int)

    # 월별 Fact 테이블 (신규!)
    fact_monthly = monthly_df[["region_id", "year_month", "value"]].copy()
    fact_monthly.rename(columns={"value": "total_pop"}, inplace=True)
    fact_monthly["total_pop"] = fact_monthly["total_pop"].astype(int)

    # 연도별 Fact 테이블 (기존 - 호환성 유지)
    monthly_df["year"] = monthly_df["year_month"].str.slice(0, 4).astype(int)
    fact_yearly = (
        monthly_df.groupby(["region_id", "year"], as_index=False)["value"]
        .mean()
        .round()
        .rename(columns={"value": "total_pop"})
    )
    fact_yearly["total_pop"] = fact_yearly["total_pop"].astype(int)

    logger.info(f"✓ 인구 데이터 정제 완료: 월별 {len(fact_monthly)}행, 연도별 {len(fact_yearly)}행")

    return fact_monthly, fact_yearly


def create_dimension_region() -> pd.DataFrame:
    """지역 차원 테이블 생성"""

    regions = pd.DataFrame([
        {"region_id": code, "region_name": name}
        for name, code in REGION_CODE_MAP.items()
    ])

    return regions


if __name__ == "__main__":
    # 테스트 실행
    unemployment = extract_unemployment()
    employment, industry = extract_employment()
    pop_monthly, pop_yearly = extract_population()
    region = create_dimension_region()

    print("\n=== ETL 결과 ===")
    print(f"실업률: {unemployment.shape}")
    print(f"고용: {employment.shape}")
    print(f"산업: {industry.shape}")
    print(f"인구(월별): {pop_monthly.shape}")
    print(f"인구(연도별): {pop_yearly.shape}")
    print(f"지역: {region.shape}")
