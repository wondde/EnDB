"""
SQL 기반 분석기

역할: 임베디드 SQL을 실행하여 유의미한 인사이트 도출
"""

import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = PROJECT_ROOT / "sql"


def execute_query_from_file(engine: Engine, sql_file: Path, query_name: str) -> pd.DataFrame:
    """SQL 파일에서 특정 쿼리를 읽어서 실행 (임베디드 SQL)"""

    with open(sql_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 쿼리 이름으로 쿼리 추출
    queries = {}
    current_query_name = None
    current_query_lines = []

    for line in content.split("\n"):
        if line.strip().startswith("-- [") and "]" in line:
            # 새로운 쿼리 시작
            if current_query_name and current_query_lines:
                queries[current_query_name] = "\n".join(current_query_lines)
            current_query_name = line.split("[")[1].split("]")[0].strip()
            current_query_lines = []
        elif current_query_name and not line.strip().startswith("--"):
            current_query_lines.append(line)

    # 마지막 쿼리 저장
    if current_query_name and current_query_lines:
        queries[current_query_name] = "\n".join(current_query_lines)

    if query_name not in queries:
        raise ValueError(f"쿼리 '{query_name}'를 찾을 수 없습니다.")

    query = queries[query_name].strip().rstrip(";")

    with engine.connect() as conn:
        result = pd.read_sql_query(text(query), conn)

    logger.info(f"✓ SQL 실행: {query_name} ({len(result)}행)")
    return result


def run_all_insights(engine: Engine) -> Dict[str, pd.DataFrame]:
    """모든 인사이트 쿼리 실행"""

    insights = {}
    # SQLite용 쿼리 파일 사용
    sql_file = SQL_DIR / "insights_sqlite.sql"

    insight_names = [
        "인사이트 1",  # 실업률 감소에 기여한 산업
        "인사이트 2",  # 실업률 변동성
        "인사이트 3",  # 산업 다각화 지수
        "인사이트 4",  # 고용 회복력
        "인사이트 5",  # 경제활동참가율 변화
    ]

    for name in insight_names:
        try:
            insights[name] = execute_query_from_file(engine, sql_file, name)
        except Exception as e:
            logger.error(f"✗ {name} 실행 실패: {e}")

    return insights


def print_insights(insights: Dict[str, pd.DataFrame]) -> None:
    """인사이트 결과를 보기 좋게 출력"""

    print("\n" + "=" * 80)
    print("📊 노동시장 데이터 분석 결과 (임베디드 SQL 기반)")
    print("=" * 80 + "\n")

    for i, (name, df) in enumerate(insights.items(), 1):
        print(f"[{i}] {name}")
        print("-" * 80)
        print(df.to_string(index=False))
        print()

        # 간단한 해석 추가
        if "인사이트 1" in name and len(df) > 0:
            print(f"💡 {df.iloc[0]['industry_name']}이(가) 가장 많은 고용을 창출했습니다.")
            print(f"   총 {df.iloc[0]['total_employment_change']:,.0f}명 증가\n")

        elif "인사이트 2" in name and len(df) > 0:
            volatile_regions = df[df['volatility_level'] == '높음']
            if len(volatile_regions) > 0:
                print(f"💡 {len(volatile_regions)}개 지역이 높은 실업률 변동성을 보입니다.")
                print(f"   가장 불안정: {volatile_regions.iloc[0]['region_name']}\n")

        elif "인사이트 3" in name and len(df) > 0:
            most_diverse = df.iloc[0]
            print(f"💡 가장 다각화된 지역: {most_diverse['region_name']}")
            print(f"   다각화 지수: {most_diverse['diversification_index']}\n")

        elif "인사이트 4" in name and len(df) > 0:
            best_recovery = df.iloc[0]
            print(f"💡 고용 회복이 가장 강한 지역: {best_recovery['region_name']}")
            print(f"   회복률: {best_recovery['recovery_rate_pct']}%\n")

        elif "인사이트 5" in name and len(df) > 0:
            top_increase = df.iloc[0]
            print(f"💡 경제활동참가율 증가 1위: {top_increase['region_name']}")
            print(f"   증가폭: {top_increase['rate_change']}%p\n")

    print("=" * 80)
    print("✅ 분석 완료!")
    print("=" * 80 + "\n")


def run_basic_statistics(engine: Engine) -> None:
    """기본 통계 요약"""

    print("\n" + "=" * 80)
    print("📈 기본 통계")
    print("=" * 80 + "\n")

    # 전체 데이터 개수
    with engine.connect() as conn:
        stats = pd.read_sql_query(text("""
            SELECT
                (SELECT COUNT(*) FROM fact_unemployment_monthly) as unemployment_rows,
                (SELECT COUNT(*) FROM fact_employment_by_industry_monthly) as employment_rows,
                (SELECT COUNT(*) FROM dim_industry) as industries,
                (SELECT COUNT(*) FROM dim_region) as regions
        """), conn)

    print("데이터 현황:")
    print(f"  - 실업률 데이터: {stats['unemployment_rows'][0]:,}행")
    print(f"  - 고용 데이터: {stats['employment_rows'][0]:,}행")
    print(f"  - 산업 수: {stats['industries'][0]}개")
    print(f"  - 지역 수: {stats['regions'][0]}개\n")


if __name__ == "__main__":
    from db_loader import DBConfig

    # 설정 (SQLite - 비밀번호 불필요!)
    config = DBConfig()
    engine = config.make_engine()

    # 기본 통계
    run_basic_statistics(engine)

    # 인사이트 분석
    insights = run_all_insights(engine)
    print_insights(insights)
