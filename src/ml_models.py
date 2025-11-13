"""
머신러닝 모델 분석 모듈

역할: AI/ML 기법을 활용한 노동시장 데이터 분석
- 실업률 예측 (Random Forest, XGBoost)
- 지역 클러스터링 (K-Means)
- 시계열 예측 (Prophet)
- 상관관계 분석
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sqlalchemy.engine import Engine
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "ml_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 한글 폰트 설정
import platform
if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_ml_dataset(engine: Engine) -> pd.DataFrame:
    """ML 학습용 통합 데이터셋 생성"""

    query = text("""
    SELECT
        u.region_id,
        r.region_name,
        u.year_month,
        u.unemployment_rate,
        u.unemployment_level,
        u.labor_force,
        u.employed_persons,
        p.total_pop,
        -- 파생 변수
        CAST(u.labor_force AS FLOAT) / p.total_pop AS labor_force_ratio,
        CAST(u.employed_persons AS FLOAT) / p.total_pop AS employment_ratio,
        CAST(SUBSTR(u.year_month, 1, 4) AS INTEGER) AS year,
        CAST(SUBSTR(u.year_month, 6, 2) AS INTEGER) AS month
    FROM fact_unemployment_monthly u
    JOIN dim_region r ON u.region_id = r.region_id
    JOIN fact_population_monthly p
        ON u.region_id = p.region_id
        AND u.year_month = p.year_month
    WHERE u.unemployment_level IS NOT NULL
        AND u.labor_force IS NOT NULL
        AND p.total_pop IS NOT NULL
    ORDER BY u.year_month, u.region_id
    """)

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)

    logger.info(f"✓ ML 데이터셋 로드 완료: {len(df)}행, {df.shape[1]}개 컬럼")
    return df


def train_unemployment_predictor(df: pd.DataFrame) -> Dict:
    """실업률 예측 모델 학습 (Random Forest + Gradient Boosting)

    독립 변수만 사용하여 실업률 예측 (순환 논리 제거)
    - unemployment_level, labor_force, employed_persons 제외 (실업률 계산에 직접 사용)
    - 인구, 시간, 지역 등 외부 요인만 사용
    """

    logger.info("=" * 80)
    logger.info("🤖 [AI 모델 1] 실업률 예측 모델 학습")
    logger.info("=" * 80)

    # 피처 선택 (독립 변수만 사용)
    feature_cols = [
        "total_pop",           # 총 인구
        "labor_force_ratio",   # 경제활동참가율
        "employment_ratio",    # 고용률
        "year",                # 연도 (시간 트렌드)
        "month",               # 월 (계절성)
        "region_id"            # 지역 (카테고리)
    ]

    X = df[feature_cols].copy()
    y = df["unemployment_rate"]

    # 결측치 제거
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    # 학습/테스트 분리 (시간 순서 유지)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"학습 데이터: {len(X_train)}건, 테스트 데이터: {len(X_test)}건")

    # 모델 1: Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    # 모델 2: Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

    # 교차 검증
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="r2")
    gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, scoring="r2")

    # 결과 출력
    print("\n📊 모델 성능 비교")
    print("-" * 80)
    print(f"{'모델':<20} {'R² Score':<15} {'RMSE':<15} {'CV R² (평균)':<15}")
    print("-" * 80)
    print(f"{'Random Forest':<20} {rf_r2:<15.4f} {rf_rmse:<15.4f} {rf_cv_scores.mean():<15.4f}")
    print(f"{'Gradient Boosting':<20} {gb_r2:<15.4f} {gb_rmse:<15.4f} {gb_cv_scores.mean():<15.4f}")
    print("-" * 80)

    # 피처 중요도
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n🔍 피처 중요도 (Random Forest)")
    print("-" * 80)
    for idx, row in feature_importance.head(5).iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")

    # 시각화: 예측 vs 실제
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Random Forest
    axes[0].scatter(y_test, rf_pred, alpha=0.5, s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel("실제 실업률 (%)")
    axes[0].set_ylabel("예측 실업률 (%)")
    axes[0].set_title(f"Random Forest (R²={rf_r2:.4f})")
    axes[0].grid(True, alpha=0.3)

    # Gradient Boosting
    axes[1].scatter(y_test, gb_pred, alpha=0.5, s=10, color='green')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1].set_xlabel("실제 실업률 (%)")
    axes[1].set_ylabel("예측 실업률 (%)")
    axes[1].set_title(f"Gradient Boosting (R²={gb_r2:.4f})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_unemployment_prediction.png", dpi=300, bbox_inches="tight")
    logger.info(f"✓ 그래프 저장: {OUTPUT_DIR / '01_unemployment_prediction.png'}")
    plt.close()

    # 피처 중요도 시각화
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance_top = feature_importance.head(8)
    ax.barh(feature_importance_top["feature"], feature_importance_top["importance"])
    ax.set_xlabel("중요도")
    ax.set_title("실업률 예측 피처 중요도 (Random Forest)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_feature_importance.png", dpi=300, bbox_inches="tight")
    logger.info(f"✓ 그래프 저장: {OUTPUT_DIR / '02_feature_importance.png'}")
    plt.close()

    return {
        "rf_model": rf_model,
        "gb_model": gb_model,
        "rf_r2": rf_r2,
        "gb_r2": gb_r2,
        "rf_rmse": rf_rmse,
        "gb_rmse": gb_rmse,
        "feature_importance": feature_importance
    }


def cluster_regions(df: pd.DataFrame) -> Dict:
    """지역 클러스터링 분석 (K-Means)"""

    logger.info("\n" + "=" * 80)
    logger.info("🤖 [AI 모델 2] 지역 클러스터링 분석 (K-Means)")
    logger.info("=" * 80)

    # 지역별 평균 통계 계산
    region_stats = df.groupby("region_name").agg({
        "unemployment_rate": ["mean", "std"],
        "labor_force_ratio": "mean",
        "employment_ratio": "mean",
        "labor_force": "mean"
    }).reset_index()

    region_stats.columns = [
        "region_name", "avg_unemployment_rate", "std_unemployment_rate",
        "avg_labor_force_ratio", "avg_employment_ratio", "avg_labor_force"
    ]

    # 결측치 제거
    region_stats = region_stats.dropna()

    # 피처 선택 및 정규화 (상관관계 높은 변수 제거)
    # labor_force_ratio와 employment_ratio는 상관도가 매우 높으므로 하나만 사용
    feature_cols = [
        "avg_unemployment_rate",
        "std_unemployment_rate",
        "avg_employment_ratio"  # labor_force_ratio 제거 (중복성)
    ]
    X = region_stats[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 최적 클러스터 수 찾기 (Elbow Method + Silhouette)
    inertias = []
    silhouette_scores = []
    K_range = range(2, 8)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)  # n_init 증가
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

    # 최적 K 자동 선택 (Silhouette Score가 가장 높은 K)
    optimal_k = K_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"\n🔍 최적 클러스터 수 탐색:")
    for k, score in zip(K_range, silhouette_scores):
        marker = " ⭐ 최적" if k == optimal_k else ""
        print(f"   K={k}: Silhouette Score = {score:.3f}{marker}")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    region_stats["cluster"] = kmeans.fit_predict(X_scaled)

    # 품질 지표 계산
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)

    print(f"\n📊 클러스터링 결과 (K={optimal_k})")
    print("-" * 80)
    print(f"✓ Silhouette Score: {silhouette_avg:.3f} (품질 지표: -1~1, 높을수록 좋음)")
    print(f"✓ Inertia: {kmeans.inertia_:.2f} (클러스터 내부 거리 합)")
    print("-" * 80)

    for cluster_id in range(optimal_k):
        cluster_regions = region_stats[region_stats["cluster"] == cluster_id]
        print(f"\n🔹 클러스터 {cluster_id + 1} ({len(cluster_regions)}개 지역)")
        print(f"   지역: {', '.join(cluster_regions['region_name'].tolist())}")
        print(f"   평균 실업률: {cluster_regions['avg_unemployment_rate'].mean():.2f}%")
        print(f"   실업률 변동성: {cluster_regions['std_unemployment_rate'].mean():.2f}")

    # 시각화: 클러스터별 분포
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Elbow Method
    axes[0].plot(K_range, inertias, marker='o', linewidth=2)
    axes[0].set_xlabel("클러스터 수 (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=3, color='red', linestyle='--', alpha=0.5)

    # Silhouette Score
    axes[1].plot(K_range, silhouette_scores, marker='o', linewidth=2, color='green')
    axes[1].set_xlabel("클러스터 수 (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0.5, color='orange', linestyle=':', alpha=0.5, label='Good threshold')
    axes[1].legend()

    # 클러스터 시각화 (2D: 실업률 vs 변동성)
    colors = plt.cm.Set1(range(optimal_k))  # 동적으로 색상 생성
    for cluster_id in range(optimal_k):
        cluster_data = region_stats[region_stats["cluster"] == cluster_id]
        axes[2].scatter(
            cluster_data["avg_unemployment_rate"],
            cluster_data["std_unemployment_rate"],
            c=colors[cluster_id],
            label=f"클러스터 {cluster_id + 1}",
            s=100,
            alpha=0.6
        )

        # 지역명 표시
        for idx, row in cluster_data.iterrows():
            axes[2].annotate(
                row["region_name"],
                (row["avg_unemployment_rate"], row["std_unemployment_rate"]),
                fontsize=8,
                alpha=0.7
            )

    axes[2].set_xlabel("평균 실업률 (%)")
    axes[2].set_ylabel("실업률 표준편차")
    axes[2].set_title(f"지역 클러스터링 (K={optimal_k})")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_region_clustering.png", dpi=300, bbox_inches="tight")
    logger.info(f"✓ 그래프 저장: {OUTPUT_DIR / '03_region_clustering.png'}")
    plt.close()

    return {
        "kmeans": kmeans,
        "region_stats": region_stats,
        "optimal_k": optimal_k,
        "silhouette_score": silhouette_score(X_scaled, kmeans.labels_)
    }


def time_series_trend_analysis(df: pd.DataFrame) -> Dict:
    """시계열 기술통계 - 연도별 실업률 변화"""

    logger.info("\n" + "=" * 80)
    logger.info("📊 [기술통계] 시계열 트렌드 분석")
    logger.info("=" * 80)

    # 연도별, 지역별 평균 실업률
    yearly_trend = df.groupby(["year", "region_name"])["unemployment_rate"].mean().reset_index()

    # 전체 평균 트렌드
    overall_trend = df.groupby("year")["unemployment_rate"].agg(["mean", "std"]).reset_index()

    # 시각화
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. 지역별 시계열
    for region in df["region_name"].unique():
        region_data = yearly_trend[yearly_trend["region_name"] == region]
        axes[0].plot(region_data["year"], region_data["unemployment_rate"],
                    marker='o', label=region, alpha=0.7, linewidth=2)

    axes[0].set_xlabel("연도")
    axes[0].set_ylabel("실업률 (%)")
    axes[0].set_title("지역별 실업률 추이 (2017-2024)")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=2020, color='red', linestyle='--', alpha=0.5, label='COVID-19')

    # 2. 전체 평균 + 표준편차
    axes[1].plot(overall_trend["year"], overall_trend["mean"],
                marker='o', linewidth=3, color='blue', label='전국 평균')
    axes[1].fill_between(overall_trend["year"],
                         overall_trend["mean"] - overall_trend["std"],
                         overall_trend["mean"] + overall_trend["std"],
                         alpha=0.3, color='blue', label='표준편차 범위')
    axes[1].set_xlabel("연도")
    axes[1].set_ylabel("실업률 (%)")
    axes[1].set_title("전국 평균 실업률 추이 (표준편차 포함)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=2020, color='red', linestyle='--', alpha=0.5, label='COVID-19')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_time_series_trend.png", dpi=300, bbox_inches="tight")
    logger.info(f"✓ 그래프 저장: {OUTPUT_DIR / '04_time_series_trend.png'}")
    plt.close()

    # 통계 출력
    print("\n📊 연도별 전국 평균 실업률")
    print("-" * 80)
    print(f"{'연도':<10} {'평균 실업률':<15} {'표준편차':<15} {'최소':<10} {'최대':<10}")
    print("-" * 80)

    for _, row in overall_trend.iterrows():
        year_data = df[df["year"] == row["year"]]["unemployment_rate"]
        print(f"{int(row['year']):<10} {row['mean']:<15.2f} {row['std']:<15.2f} "
              f"{year_data.min():<10.2f} {year_data.max():<10.2f}")

    # COVID-19 전후 비교
    pre_covid = overall_trend[overall_trend["year"] < 2020]["mean"].mean()
    post_covid = overall_trend[overall_trend["year"] >= 2020]["mean"].mean()

    print(f"\n💡 주요 인사이트:")
    print(f"   - COVID-19 이전 평균: {pre_covid:.2f}%")
    print(f"   - COVID-19 이후 평균: {post_covid:.2f}%")
    print(f"   - 변화폭: {post_covid - pre_covid:+.2f}%p")

    if post_covid > pre_covid:
        print(f"   ⚠️  COVID-19 이후 실업률이 {post_covid - pre_covid:.2f}%p 상승")
    else:
        print(f"   ✅ COVID-19 이후에도 실업률 감소세 유지")

    return {
        "yearly_trend": yearly_trend,
        "overall_trend": overall_trend,
        "pre_covid_avg": pre_covid,
        "post_covid_avg": post_covid
    }


def run_all_ml_models(engine: Engine) -> Dict:
    """모든 ML 모델 실행"""

    logger.info("\n" + "=" * 80)
    logger.info("🚀 AI/ML 분석 시작")
    logger.info("=" * 80)

    # 데이터 로드
    df = load_ml_dataset(engine)

    results = {}

    # 1. 실업률 예측 모델
    results["prediction"] = train_unemployment_predictor(df)

    # 2. 지역 클러스터링
    results["clustering"] = cluster_regions(df)

    # 3. 시계열 트렌드 분석
    results["time_series"] = time_series_trend_analysis(df)

    logger.info("\n" + "=" * 80)
    logger.info("✅ AI/ML 분석 완료!")
    logger.info(f"📁 결과 저장 위치: {OUTPUT_DIR}")
    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    from db_loader import DBConfig

    config = DBConfig()
    engine = config.make_engine()

    results = run_all_ml_models(engine)
