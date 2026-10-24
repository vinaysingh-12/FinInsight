"""Structured analysis of consumer complaints (CFPB-style exports).

Produces Power BI-ready star-schema style outputs:
  fact:  complaints_clean.csv                (row level)
  dim:   dim_date.csv
  aggs:  complaint_monthly_trends.csv, complaints_by_product.csv, top_complaint_issues.csv,
         company_response_summary.csv, company_scorecard.csv, monthly_by_product.csv
  qa:    data_quality_report.json
  plots: PNG charts for the README / resume
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import OUTPUT_DIR, ensure_dirs, get_logger

log = get_logger("complaint_analysis")

# Candidate column names across CFPB export versions (matched case-insensitively).
CANDIDATES = {
    "date": ["date received", "date_received", "received date", "date"],
    "product": ["product"],
    "issue": ["issue"],
    "company": ["company"],
    "response": ["company response to consumer", "company_response_to_consumer", "response"],
    "timely": ["timely response?", "timely response", "timely_response"],
    "state": ["state"],
    "complaint_id": ["complaint id", "complaint_id"],
}
RELIEF_PREFIXES = ("closed with monetary relief", "closed with non-monetary relief")


def detect_columns(df: pd.DataFrame, overrides: dict[str, str | None]) -> dict[str, str | None]:
    lower = {c.lower().strip(): c for c in df.columns}
    mapping: dict[str, str | None] = {}
    for key, names in CANDIDATES.items():
        if overrides.get(key):
            if overrides[key] not in df.columns:
                raise SystemExit(f"--{key}-col '{overrides[key]}' not in file. Columns: {list(df.columns)}")
            mapping[key] = overrides[key]
        else:
            mapping[key] = next((lower[n] for n in names if n in lower), None)
    required = ["date", "product", "issue", "company", "response"]
    missing = [k for k in required if not mapping[k]]
    if missing:
        raise SystemExit(f"Could not detect columns for {missing}. Pass --<name>-col. "
                         f"Columns found: {list(df.columns)}")
    return mapping


def clean(df: pd.DataFrame, m: dict[str, str | None]) -> tuple[pd.DataFrame, dict]:
    qa = {"rows_in": int(len(df))}
    out = pd.DataFrame({
        "date_received": pd.to_datetime(df[m["date"]], errors="coerce"),
        "product": df[m["product"]],
        "issue": df[m["issue"]],
        "company": df[m["company"]],
        "response": df[m["response"]],
    })
    for key in ("timely", "state", "complaint_id"):
        if m[key]:
            out[key] = df[m[key]]

    qa["unparseable_dates"] = int(out["date_received"].isna().sum())
    for c in ("product", "issue", "company", "response"):
        out[c] = out[c].astype("string").str.strip().replace("", pd.NA)
    qa["missing_values"] = {c: int(out[c].isna().sum()) for c in out.columns}

    out = out.dropna(subset=["date_received", "product", "issue", "company"])
    out["response"] = out["response"].fillna("Unknown")
    if "complaint_id" in out:
        n0 = len(out)
        out = out.drop_duplicates(subset="complaint_id")
        qa["duplicate_ids_dropped"] = int(n0 - len(out))

    out["year_month"] = out["date_received"].dt.to_period("M").dt.to_timestamp()
    out["got_relief"] = out["response"].str.lower().str.startswith(RELIEF_PREFIXES).astype(int)
    if "timely" in out:
        out["timely_flag"] = out["timely"].astype("string").str.strip().str.lower().eq("yes").astype(int)

    qa["rows_out"] = int(len(out))
    qa["rows_dropped"] = qa["rows_in"] - qa["rows_out"]
    qa["date_min"] = str(out["date_received"].min().date())
    qa["date_max"] = str(out["date_received"].max().date())
    return out.reset_index(drop=True), qa


def build_dim_date(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.date_range(start.normalize(), end.normalize(), freq="D")})
    d["year"] = d["date"].dt.year
    d["quarter"] = "Q" + d["date"].dt.quarter.astype(str)
    d["month_num"] = d["date"].dt.month
    d["month_name"] = d["date"].dt.strftime("%b")
    d["year_month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    d["week_start"] = d["date"] - pd.to_timedelta(d["date"].dt.weekday, unit="D")
    d["is_weekend"] = (d["date"].dt.weekday >= 5).astype(int)
    return d


def aggregate(df: pd.DataFrame, top_n: int) -> dict[str, pd.DataFrame]:
    total = len(df)
    monthly = (df.groupby("year_month").size().rename("complaint_count").reset_index())
    monthly["mom_change_pct"] = (monthly["complaint_count"].pct_change() * 100).round(1)
    monthly["rolling_3m_avg"] = monthly["complaint_count"].rolling(3).mean().round(1)

    by_product = df.groupby("product").agg(complaint_count=("product", "size"),
                                           relief_rate=("got_relief", "mean")).reset_index()
    by_product["share_pct"] = (by_product["complaint_count"] / total * 100).round(1)
    by_product["relief_rate"] = (by_product["relief_rate"] * 100).round(1)
    by_product = by_product.sort_values("complaint_count", ascending=False)

    issues = (df.groupby(["product", "issue"]).size().rename("complaint_count").reset_index()
              .sort_values("complaint_count", ascending=False))
    issues["share_pct"] = (issues["complaint_count"] / total * 100).round(1)
    issues["rank"] = range(1, len(issues) + 1)
    issues = issues.head(top_n)

    resp = df.groupby(["company", "response"]).size().rename("complaint_count").reset_index()
    resp["share_of_company_pct"] = (resp["complaint_count"]
                                    / resp.groupby("company")["complaint_count"].transform("sum") * 100).round(1)

    agg = {"got_relief": "mean"}
    score = df.groupby("company").agg(total_complaints=("company", "size"),
                                      relief_rate_pct=("got_relief", lambda s: round(s.mean() * 100, 1)))
    if "timely_flag" in df:
        score["timely_response_rate_pct"] = df.groupby("company")["timely_flag"].mean().mul(100).round(1)
    score = score.reset_index().sort_values("total_complaints", ascending=False)

    mbp = df.groupby(["year_month", "product"]).size().rename("complaint_count").reset_index()

    return {"complaint_monthly_trends": monthly, "complaints_by_product": by_product,
            "top_complaint_issues": issues, "company_response_summary": resp,
            "company_scorecard": score, "monthly_by_product": mbp}


def plots(tables: dict[str, pd.DataFrame]) -> None:
    blue, grey = "#2a6fb0", "#9aa5b1"
    m = tables["complaint_monthly_trends"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(m["year_month"], m["complaint_count"], width=20, color=grey, label="Monthly complaints")
    ax.plot(m["year_month"], m["rolling_3m_avg"], color=blue, lw=2.2, label="3-month average")
    ax.set_title("Complaints per month")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_monthly_trend.png", dpi=160)
    plt.close(fig)

    p = tables["complaints_by_product"].sort_values("complaint_count")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(p["product"], p["complaint_count"], color=blue)
    ax.set_title("Complaints by product")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_by_product.png", dpi=160)
    plt.close(fig)

    t = tables["top_complaint_issues"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(t["issue"] + "  (" + t["product"] + ")", t["complaint_count"], color=blue)
    ax.set_title("Top 10 complaint issues")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chart_top_issues.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--top-n", type=int, default=15)
    for key in CANDIDATES:
        ap.add_argument(f"--{key}-col", default=None, help=f"Override detected '{key}' column")
    args = ap.parse_args()

    ensure_dirs()
    raw = pd.read_csv(args.input, low_memory=False)
    mapping = detect_columns(raw, {k: getattr(args, f"{k}_col") for k in CANDIDATES})
    log.info("Column mapping: %s", {k: v for k, v in mapping.items() if v})

    df, qa = clean(raw, mapping)
    log.info("Cleaned %d -> %d rows (%s to %s)", qa["rows_in"], qa["rows_out"], qa["date_min"], qa["date_max"])
    qa["input_file"] = str(args.input)
    qa["column_mapping"] = {k: v for k, v in mapping.items() if v}

    tables = aggregate(df, args.top_n)
    df.to_csv(OUTPUT_DIR / "complaints_clean.csv", index=False)
    build_dim_date(df["date_received"].min(), df["date_received"].max()).to_csv(
        OUTPUT_DIR / "dim_date.csv", index=False)
    for name, t in tables.items():
        t.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    (OUTPUT_DIR / "data_quality_report.json").write_text(json.dumps(qa, indent=2, default=str))
    plots(tables)

    top = tables["complaints_by_product"].iloc[0]
    log.info("Top product: %s (%s complaints, %.1f%%)", top["product"], top["complaint_count"], top["share_pct"])
    log.info("Wrote %d tables + charts to %s", len(tables) + 2, OUTPUT_DIR)


if __name__ == "__main__":
    main()
