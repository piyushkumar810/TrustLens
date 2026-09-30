from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_and_prepare_dataframe() -> pd.DataFrame:
    listings = pd.read_csv(DATA_DIR / "listings.csv")
    listings = listings.drop_duplicates().copy()

    # Convert timestamp-like fields
    if "event_time" in listings.columns:
        listings["event_time"] = pd.to_datetime(listings["event_time"], errors="coerce")

    # Fill missing values
    numeric_cols = listings.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        listings[col] = listings[col].fillna(listings[col].median())

    text_cols = listings.select_dtypes(exclude=[np.number]).columns
    for col in text_cols:
        if listings[col].isna().any():
            mode = listings[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            listings[col] = listings[col].fillna(fill_value)

    # IQR-based outlier removal on price
    if "price" in listings.columns:
        q1 = listings["price"].quantile(0.25)
        q3 = listings["price"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        listings = listings[(listings["price"] >= lower) & (listings["price"] <= upper)].copy()

    # Z-score based cleanup on price for heavy outliers
    if "price" in listings.columns:
        z_scores = (listings["price"] - listings["price"].mean()) / listings["price"].std(ddof=0)
        listings = listings[z_scores.abs() <= 3].copy()

    # Combine synthetic anomaly columns into one target for supervised preprocessing work
    target_cols = ["price_anomaly", "mismatch", "duplicate", "seller_bad"]
    for col in target_cols:
        if col not in listings.columns:
            raise ValueError(f"Missing required column: {col}")
    listings["fraud_target"] = listings[target_cols].max(axis=1).astype(int)

    # Save cleaned dataset
    cleaned_path = DATA_DIR / "listings_clean.csv"
    listings.to_csv(cleaned_path, index=False)
    print(f"Cleaned dataframe saved to: {cleaned_path}")
    print(f"Rows after cleaning: {len(listings)}")
    print(f"Missing values after cleaning: {listings.isna().sum().sum()}")
    print(listings[["price_anomaly", "mismatch", "duplicate", "seller_bad", "fraud_target"]].mean().to_dict())
    return listings


def create_eda_plots(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    df["price"].hist(bins=30, edgecolor="black")
    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    df["fraud_target"].value_counts().sort_index().plot(kind="bar", color=["#4C72B0", "#DD8452"])
    plt.title("Fraud Target Balance")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.tight_layout()
    plot_path = REPORTS_DIR / "step4_eda.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved EDA plot to: {plot_path}")


def create_train_test_split(df: pd.DataFrame) -> None:
    target_col = "fraud_target"
    feature_cols = [c for c in df.columns if c not in {"product_id", "event_time", target_col}]
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    cat_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=list(cat_cols), drop_first=False)

    # Scale numeric columns after one-hot encoding
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    split_dir = DATA_DIR / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(split_dir / "X_train.csv", index=False)
    X_test.to_csv(split_dir / "X_test.csv", index=False)
    y_train.to_csv(split_dir / "y_train.csv", index=False, header=["fraud_target"])
    y_test.to_csv(split_dir / "y_test.csv", index=False, header=["fraud_target"])

    print(f"Saved train/test split to: {split_dir}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}")


if __name__ == "__main__":
    df = clean_and_prepare_dataframe()
    create_eda_plots(df)
    create_train_test_split(df)
