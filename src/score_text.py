"""Score new financial text with a trained model and export Power BI-ready predictions."""
from __future__ import annotations

import argparse

import joblib
import pandas as pd

from utils import MODEL_DIR, OUTPUT_DIR, ensure_dirs, get_logger

log = get_logger("score_text")


def score(df: pd.DataFrame, bundle: dict, text_col: str = "text",
          review_threshold: float = 0.60) -> pd.DataFrame:
    pipe = bundle["pipeline"]
    texts = df[text_col].fillna("").astype(str)
    proba = pipe.predict_proba(texts)
    out = df.copy()
    out["predicted_sentiment"] = pipe.classes_[proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1).round(4)
    for i, cls in enumerate(pipe.classes_):
        out[f"prob_{cls}"] = proba[:, i].round(4)
    # Confidence is NOT a calibrated probability; use it only to prioritise human review.
    out["needs_review"] = out["confidence"] < review_threshold
    out["model_name"] = bundle["model_name"]
    out["scored_at_utc"] = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="CSV with a text column")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--model", default=str(MODEL_DIR / "sentiment_model.joblib"))
    ap.add_argument("--output", default=str(OUTPUT_DIR / "sentiment_predictions.csv"))
    ap.add_argument("--review-threshold", type=float, default=0.60)
    args = ap.parse_args()

    ensure_dirs()
    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise SystemExit(f"Column '{args.text_col}' not found. Available: {list(df.columns)}")
    bundle = joblib.load(args.model)
    out = score(df, bundle, args.text_col, args.review_threshold)
    out.to_csv(args.output, index=False)
    log.info("Scored %d rows -> %s | distribution: %s", len(out), args.output,
             out["predicted_sentiment"].value_counts().to_dict())


if __name__ == "__main__":
    main()
