"""
DB Loader 모듈

역할: DataFrame을 SQLite DB에 적재 (임베디드 SQL 사용)
"""

from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = PROJECT_ROOT / "sql"
DB_DIR = PROJECT_ROOT / "data"


@dataclass
class DBConfig:
    """SQLite 접속 정보"""
    db_path: str = "data/employment.db"

    def make_engine(self) -> Engine:
        # 프로젝트 루트 기준 상대 경로로 DB 파일 생성
        db_file = PROJECT_ROOT / self.db_path
        db_file.parent.mkdir(parents=True, exist_ok=True)

        uri = f"sqlite:///{db_file}"
        logger.info(f"SQLite DB 경로: {db_file}")
        return create_engine(uri, echo=False)


def execute_sql_file(engine: Engine, sql_file: Path) -> None:
    """SQL 파일 실행 (임베디드 SQL)"""

    if not sql_file.exists():
        logger.warning(f"SQL 파일 없음: {sql_file}")
        return

    with open(sql_file, "r", encoding="utf-8") as f:
        sql_content = f.read()

    # 세미콜론 기준으로 쿼리 분리
    statements = [stmt.strip() for stmt in sql_content.split(";") if stmt.strip()]

    with engine.connect() as conn:
        for i, stmt in enumerate(statements, 1):
            try:
                conn.execute(text(stmt))
                conn.commit()
                logger.info(f"✓ SQL 실행 완료: {sql_file.name} ({i}/{len(statements)})")
            except Exception as e:
                # 테이블이 이미 존재하는 경우 등은 무시
                if "already exists" not in str(e).lower():
                    logger.error(f"✗ SQL 실행 실패: {e}")


def load_to_database(
    engine: Engine,
    unemployment: pd.DataFrame,
    employment: pd.DataFrame,
    industry: pd.DataFrame,
    pop_monthly: pd.DataFrame,
    pop_yearly: pd.DataFrame,
    region: pd.DataFrame,
) -> None:
    """데이터를 SQLite에 적재"""

    # 1. 테이블 생성
    logger.info("1. 테이블 생성 중...")
    execute_sql_file(engine, SQL_DIR / "create_tables_sqlite.sql")

    # 2. 기존 데이터 삭제 (SQLite는 TRUNCATE 없음, DELETE 사용)
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        conn.execute(text("DELETE FROM fact_unemployment_monthly"))
        conn.execute(text("DELETE FROM fact_employment_by_industry_monthly"))
        conn.execute(text("DELETE FROM fact_population_monthly"))
        conn.execute(text("DELETE FROM fact_population_yearly"))
        conn.execute(text("DELETE FROM dim_industry"))
        conn.execute(text("DELETE FROM dim_region"))
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()
    logger.info("✓ 기존 데이터 삭제 완료")

    # 3. Dimension 테이블 먼저 적재
    region.to_sql("dim_region", engine, if_exists="append", index=False)
    industry.to_sql("dim_industry", engine, if_exists="append", index=False)
    logger.info(f"✓ Dimension 적재: 지역 {len(region)}건, 산업 {len(industry)}건")

    # 4. Fact 테이블 적재
    unemployment.to_sql("fact_unemployment_monthly", engine, if_exists="append", index=False)
    employment.to_sql("fact_employment_by_industry_monthly", engine, if_exists="append", index=False)
    pop_monthly.to_sql("fact_population_monthly", engine, if_exists="append", index=False)
    pop_yearly.to_sql("fact_population_yearly", engine, if_exists="append", index=False)
    logger.info(f"✓ Fact 적재: 실업률 {len(unemployment)}건, 고용 {len(employment)}건, "
                f"인구(월별) {len(pop_monthly)}건, 인구(연도별) {len(pop_yearly)}건")

    logger.info("=" * 60)
    logger.info("✅ DB 적재 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    # 테스트용
    from etl import (
        extract_unemployment,
        extract_employment,
        extract_population,
        create_dimension_region
    )

    config = DBConfig()  # 비밀번호 불필요!
    engine = config.make_engine()

    unemployment = extract_unemployment()
    employment, industry = extract_employment()
    pop_monthly, pop_yearly = extract_population()
    region = create_dimension_region()

    load_to_database(engine, unemployment, employment, industry, pop_monthly, pop_yearly, region)
