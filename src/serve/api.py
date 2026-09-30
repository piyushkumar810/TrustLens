from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serve.batch_score_step9 import read_predictions

app = Flask(__name__)


@app.get("/health")
def health() -> tuple[dict, int]:
    df = read_predictions()
    return jsonify({"status": "ok", "rows": int(len(df))}), 200


@app.get("/api/listing")
def listing_by_query():
    product_id = request.args.get("product_id", type=int)
    if product_id is None:
        return jsonify({"error": "Missing product_id query parameter."}), 400
    return get_listing(product_id)


@app.get("/api/listing/<int:product_id>")
def listing_by_path(product_id: int):
    return get_listing(product_id)


def get_listing(product_id: int):
    df = read_predictions()
    match = df[df["product_id"] == product_id]
    if match.empty:
        return jsonify({"error": "Listing not found", "product_id": product_id}), 404

    row = match.iloc[0].to_dict()
    payload = {
        "product_id": int(row["product_id"]),
        "seller_id": int(row["seller_id"]),
        "category": row.get("category"),
        "price": float(row.get("price", 0.0)),
        "num_listings": int(row.get("num_listings", 0)),
        "review_count": int(row.get("review_count", 0)),
        "share_cg_reviews": float(row.get("share_cg_reviews", 0.0)),
        "return_rate": float(row.get("return_rate", 0.0)),
        "risk_score": float(row.get("risk_score", 0.0)),
        "is_flagged": int(row.get("is_flagged", 0)),
    }
    return jsonify(payload), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=False)
