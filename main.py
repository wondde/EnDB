"""
노동시장 분석 프로젝트 - 메인 실행 파일

실행 방법:
    python main.py --mode etl         # ETL만 실행
    python main.py --mode load        # DB 적재만 실행
    python main.py --mode analyze     # SQL 분석만 실행
    python main.py --mode ml          # AI/ML 분석만 실행
    python main.py --mode all         # 전체 실행 (기본값)

SQLite 사용:
    비밀번호 불필요, 파일 기반 DB (data/employment.db)
"""

import argparse
import logging
import sys
from pathlib import Path

# src 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))

from etl import (
    extract_unemployment,
    extract_employment,
    extract_population,
    create_dimension_region
)
from db_loader import DBConfig, load_to_database
from analyzer import run_all_insights, print_insights, run_basic_statistics
from ml_models import run_all_ml_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="노동시장 데이터 분석 시스템 (SQLite + AI/ML)")
    parser.add_argument(
        "--mode",
        choices=["etl", "load", "analyze", "ml", "all"],
        default="all",
        help="실행 모드 선택"
    )
    parser.add_argument(
        "--db-path",
        default="data/employment.db",
        help="SQLite DB 파일 경로 (기본값: data/employment.db)"
    )

    args = parser.parse_args()

    # DB 설정 (SQLite - 비밀번호 불필요!)
    db_config = DBConfig(db_path=args.db_path)

    try:
        print("\n" + "=" * 80)
        print("🚀 노동시장 데이터 분석 시스템")
        print("=" * 80 + "\n")

        # ETL 실행
        if args.mode in ["etl", "all"]:
            logger.info("=" * 60)
            logger.info("STEP 1: ETL (Extract, Transform, Load)")
            logger.info("=" * 60)

            unemployment = extract_unemployment()
            employment, industry = extract_employment()
            pop_monthly, pop_yearly = extract_population()
            region = create_dimension_region()

            logger.info("✅ ETL 완료\n")

        # DB 적재
        if args.mode in ["load", "all"]:
            logger.info("=" * 60)
            logger.info("STEP 2: DB 적재 (Embedded SQL)")
            logger.info("=" * 60)

            if args.mode == "load":
                # load 모드면 ETL 다시 실행
                unemployment = extract_unemployment()
                employment, industry = extract_employment()
                pop_monthly, pop_yearly = extract_population()
                region = create_dimension_region()

            engine = db_config.make_engine()
            load_to_database(engine, unemployment, employment, industry, pop_monthly, pop_yearly, region)

            logger.info("✅ DB 적재 완료\n")

        # SQL 분석 실행
        if args.mode in ["analyze", "all"]:
            logger.info("=" * 60)
            logger.info("STEP 3: 데이터 분석 (Embedded SQL)")
            logger.info("=" * 60)

            engine = db_config.make_engine()

            # 기본 통계
            run_basic_statistics(engine)

            # 인사이트 도출
            insights = run_all_insights(engine)
            print_insights(insights)

            logger.info("✅ SQL 분석 완료\n")

        # AI/ML 분석 실행
        if args.mode in ["ml", "all"]:
            logger.info("=" * 60)
            logger.info("STEP 4: AI/ML 분석")
            logger.info("=" * 60)

            engine = db_config.make_engine()
            ml_results = run_all_ml_models(engine)

            logger.info("✅ AI/ML 분석 완료\n")

        print("\n" + "=" * 80)
        print("✅ 모든 작업이 완료되었습니다!")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
