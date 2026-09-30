from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "tabular_fraud.pkl"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "price_ratio",
    "seller_age_days",
    "review_velocity",
    "share_cg_reviews",
    "return_rate",
    "num_listings",
    "avg_price",
    "median_price",
    "category_price_ratio",
    "price",
]


def read_predictions() -> pd.DataFrame:
    predictions_path = REPORTS_DIR / "step9_batch_predictions.csv"
    if not predictions_path.exists():
        score_listings()
    df = pd.read_csv(predictions_path)
    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)
    if "is_flagged" in df.columns:
        df["is_flagged"] = pd.to_numeric(df["is_flagged"], errors="coerce").fillna(0).astype(int)
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


def build_tabular_features(listings: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    seller_num_listings = listings.groupby("seller_id").size().rename("num_listings")
    seller_return_rate = (
        listings.groupby("seller_id")["price_anomaly"].mean() * 0.7
        + listings.groupby("seller_id")["duplicate"].mean() * 0.3
    ).rename("return_rate")

    review_stats = reviews.groupby("seller_id").agg(
        review_count=("review_id", "count"),
        share_cg_reviews=("label", "mean"),
    )

    listing_prices = listings.groupby("seller_id")["price"].agg(["mean", "median"])
    listing_prices.columns = ["avg_price", "median_price"]

    seller_features = pd.concat(
        [seller_num_listings, seller_return_rate, review_stats, listing_prices],
        axis=1,
    ).reset_index()

    df = listings.merge(seller_features, on="seller_id", how="left")
    df["price_ratio"] = df["price"] / df["avg_price"].replace(0, 1)
    df["seller_age_days"] = ((df["seller_id"] % 30) + 1) * 14
    df["review_velocity"] = df["review_count"] / df["seller_age_days"].clip(lower=1)
    df["category_price_ratio"] = df["price"] / df.groupby("category")["price"].transform("median")
    return df


def score_listings() -> pd.DataFrame:
    listings = pd.read_csv(DATA_DIR / "listings_clean.csv")
    reviews = pd.read_csv(DATA_DIR / "reviews.csv")
    scored_df = build_tabular_features(listings, reviews)

    model = joblib.load(MODEL_PATH)
    X = scored_df[FEATURES].to_numpy()
    probs = model.predict_proba(X)[:, 1]

    scored_df["risk_score"] = probs
    scored_df["is_flagged"] = (probs >= 0.5).astype(int)
    scored_df = scored_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    output_cols = [
        "product_id",
        "seller_id",
        "category",
        "price",
        "num_listings",
        "review_count",
        "share_cg_reviews",
        "return_rate",
        "risk_score",
        "is_flagged",
    ]

    result = scored_df[output_cols].copy()
    result.to_csv(REPORTS_DIR / "step9_batch_predictions.csv", index=False)
    return result


if __name__ == "__main__":
    predictions = score_listings()
    print(predictions.head(10).to_string(index=False))
    print(f"\nSaved {len(predictions)} scored listings to {REPORTS_DIR / 'step9_batch_predictions.csv'}")
