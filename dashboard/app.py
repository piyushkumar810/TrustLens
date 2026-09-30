from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serve.batch_score_step9 import read_predictions


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = read_predictions()
    if "risk_score" not in df.columns:
        raise ValueError("The predictions file does not contain a risk_score column.")
    return df


def main() -> None:
    st.set_page_config(page_title="TrustLens Fraud Monitor", layout="wide")
    st.title("TrustLens Fraud Monitor")
    st.caption("Marketplace fraud risk dashboard built from the saved TrustLens model.")

    df = load_predictions()

    categories = sorted(df["category"].dropna().unique().tolist())
    threshold = st.sidebar.slider("Risk threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    selected_categories = st.sidebar.multiselect("Categories", categories, default=categories)
    sellers = sorted(df["seller_id"].dropna().astype(int).unique().tolist())
    selected_seller = st.sidebar.selectbox("Seller drilldown", options=["All sellers"] + sellers, index=0)

    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df["risk_score"] >= threshold].copy()
    if selected_categories:
        filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
    if selected_seller != "All sellers":
        filtered_df = filtered_df[filtered_df["seller_id"] == selected_seller].copy()

    total_listings = len(df)
    flagged_count = int(df["is_flagged"].sum())
    avg_risk = float(df["risk_score"].mean())
    top_category = df.groupby("category")["risk_score"].mean().idxmax() if not df.empty else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Listings scored", f"{total_listings}")
    col2.metric("Flagged", f"{flagged_count}")
    col3.metric("Avg. risk", f"{avg_risk:.2%}")
    col4.metric("Highest-risk category", top_category)

    if filtered_df.empty:
        st.warning("No listings match the selected risk threshold and category filters.")
        return

    top_sellers = (
        filtered_df.groupby("seller_id", as_index=False)
        .agg(highest_risk=("risk_score", "mean"), listings=("product_id", "count"))
        .sort_values("highest_risk", ascending=False)
        .head(10)
    )

    chart_col, table_col = st.columns([1.2, 1.8])
    with chart_col:
        st.subheader("Top risky sellers")
        st.bar_chart(top_sellers.set_index("seller_id")["highest_risk"], width="stretch")

        st.subheader("Category risk overview")
        category_risk = filtered_df.groupby("category")["risk_score"].mean().sort_values(ascending=False)
        st.bar_chart(category_risk, width="stretch")

    with table_col:
        st.subheader("Flagged listings")
        display_df = filtered_df[[
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
        ]].copy()
        display_df = display_df.sort_values("risk_score", ascending=False)
        st.dataframe(display_df, width="stretch", hide_index=True)

        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export filtered CSV",
            data=csv_bytes,
            file_name="trustlens_filtered_predictions.csv",
            mime="text/csv",
        )

    if selected_seller != "All sellers":
        seller_rows = filtered_df[filtered_df["seller_id"] == selected_seller].copy()
        st.subheader(f"Seller detail for seller {selected_seller}")
        seller_summary = seller_rows[["product_id", "category", "price", "risk_score", "is_flagged"]].copy()
        seller_summary = seller_summary.sort_values("risk_score", ascending=False)
        st.dataframe(seller_summary, width="stretch", hide_index=True)
        st.metric("Seller flagged listings", int(seller_rows["is_flagged"].sum()))
        st.metric("Seller average risk", f"{seller_rows['risk_score'].mean():.2%}")

    st.caption("This dashboard reads the latest batch predictions from the TrustLens model pipeline.")


if __name__ == "__main__":
    main()
