"""Build a self-contained interactive HTML dashboard from the pipeline outputs.

Mirrors the four Power BI pages (overview, complaint deep dive, text sentiment, model
quality) with slicers and click-to-filter charts. Complaint rows are pre-aggregated by
month/product/issue/company/response, so the file stays small even for large CFPB exports.

    python src/build_dashboard.py --tag demo
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd

from utils import OUTPUT_DIR, ROOT, get_logger

log = get_logger("build_dashboard")


def read(name: str) -> pd.DataFrame | None:
    p = OUTPUT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def records(df: pd.DataFrame | None) -> list[dict]:
    if df is None:
        return []
    return json.loads(df.astype(object).where(df.notna(), None).to_json(orient="records"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="", help="Suffix used when training (e.g. 'demo')")
    ap.add_argument("--demo", action="store_true", help="Label the dashboard as synthetic demo data")
    ap.add_argument("--output", default=str(ROOT / "dashboard" / "fininsight_dashboard.html"))
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    demo = args.demo or args.tag == "demo"

    c = read("complaints_clean.csv")
    if c is None:
        raise SystemExit("outputs/complaints_clean.csv not found. Run complaint_analysis.py first.")
    has_timely = "timely_flag" in c.columns
    c["m"] = c["year_month"].astype(str).str[:7]
    c["timely_flag"] = c["timely_flag"] if has_timely else 0
    agg = (c.groupby(["m", "product", "issue", "company", "response"], observed=True)
             .agg(n=("m", "size"), rel=("got_relief", "sum"), t=("timely_flag", "sum")).reset_index()
             .rename(columns={"product": "p", "issue": "i", "company": "c", "response": "r"}))

    sent = read("sentiment_predictions.csv")
    if sent is not None:
        sent = sent[["text", "predicted_sentiment", "confidence", "needs_review"]].copy()
        sent["needs_review"] = sent["needs_review"].astype(str).str.lower().eq("true")

    mpath = OUTPUT_DIR / f"metrics{sfx}.json"
    cm = read(f"confusion_matrix{sfx}.csv")
    cls = read(f"classification_report{sfx}.csv")
    data = {
        "meta": {"demo": demo, "date_min": str(c["m"].min()), "date_max": str(c["m"].max()),
                 "complaints_source": "outputs/complaints_clean.csv", "built": date.today().isoformat(),
                 "review_threshold": 0.60},
        "has_timely": has_timely,
        "years": sorted(c["m"].str[:4].unique().tolist()),
        "products": c["product"].value_counts().index.tolist(),
        "companies": c["company"].value_counts().index.tolist(),
        "rows": records(agg),
        "sentiment": records(sent),
        "metrics": json.loads(mpath.read_text()) if mpath.exists() else None,
        "comparison": records(read(f"model_comparison{sfx}.csv")),
        "confusion": cm.iloc[:, 1:].values.tolist() if cm is not None else [],
        "classes": records(cls[cls["class"].isin(["negative", "neutral", "positive"])]) if cls is not None else [],
        "features": records(read(f"top_features{sfx}.csv")),
        "misclassified": records(read(f"misclassified_examples{sfx}.csv"))[:25],
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = (ROOT / "dashboard" / "template.html").read_text().replace("/*__DATA__*/", blob)
    out = ROOT / "dashboard" / "fininsight_dashboard.html" if args.output is None else __import__("pathlib").Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    log.info("Wrote %s (%.0f KB, %d aggregated complaint rows)", out, len(html) / 1024, len(agg))


if __name__ == "__main__":
    main()
