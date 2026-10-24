"""Generate SYNTHETIC demo data so the pipeline runs without downloads.

Nothing here is real customer data or real financial news. Do not report
metrics computed on these files as real-world performance.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from utils import DATA_DIR, RANDOM_STATE, ensure_dirs, get_logger

log = get_logger("make_demo_data")

COMPANIES = ["Northwind Bank", "Contoso Credit", "Fabrikam Finance", "Tailspin Lending",
             "Adventure Cards", "Litware Mortgage"]
PRODUCT_ISSUES = {
    "Credit card": ["Billing dispute", "Unauthorized charges", "Interest rate problems", "Rewards not credited"],
    "Mortgage": ["Escrow problems", "Payment processing", "Loan modification", "Servicing transfer"],
    "Checking account": ["Overdraft fees", "Account frozen", "Deposit not available", "Fraud alert handling"],
    "Student loan": ["Repayment plan issues", "Incorrect balance", "Payment not applied"],
    "Debt collection": ["Repeated contact attempts", "Debt not owed", "Incorrect information"],
}
RESPONSES = ["Closed with explanation", "Closed with monetary relief", "Closed with non-monetary relief",
             "In progress", "Closed without relief"]
RESP_P = [0.45, 0.12, 0.15, 0.08, 0.20]

POS = ["Revenue grew {n}% year over year, beating analyst expectations.",
       "The company raised its full-year profit guidance after a strong quarter.",
       "Operating profit increased significantly on higher demand.",
       "Shares rose after the firm announced a new record order.",
       "Net sales improved and margins expanded compared with last year."]
NEG = ["Revenue fell {n}% year over year, missing analyst expectations.",
       "The company cut its full-year profit guidance after a weak quarter.",
       "Operating loss widened due to falling demand.",
       "Shares dropped after the firm announced layoffs and a plant closure.",
       "Net sales declined and margins shrank compared with last year."]
NEU = ["The company will publish its quarterly report on {n} November.",
       "The board proposed a meeting to discuss the annual agenda.",
       "The firm operates {n} offices across the region.",
       "Management said the results will be released as scheduled.",
       "The company has {n} employees and is headquartered in the capital."]


def labeled_text(n_per_class: int = 100) -> pd.DataFrame:
    rng = random.Random(RANDOM_STATE)
    rows = []
    for label, templates in (("positive", POS), ("negative", NEG), ("neutral", NEU)):
        for _ in range(n_per_class):
            rows.append((rng.choice(templates).format(n=rng.randint(2, 9999)), label))
    df = pd.DataFrame(rows, columns=["sentence", "sentiment"]).drop_duplicates("sentence")
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def unlabeled_text() -> pd.DataFrame:
    sents = [
        "Quarterly earnings exceeded forecasts and the stock jumped.",
        "The lender reported a sharp decline in loan originations.",
        "The bank will hold its annual general meeting next month.",
        "Profit margins improved thanks to cost-cutting measures.",
        "Credit losses increased and the outlook was lowered.",
        "The company opened a new branch office in the city centre.",
        "Management expects stronger demand in the second half.",
        "Delinquencies rose, pressuring the firm's earnings.",
    ]
    return pd.DataFrame({"text_id": range(1, len(sents) + 1), "text": sents})


def complaints(n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    # mild upward trend + noise
    w = np.linspace(0.8, 1.4, len(dates)) * rng.uniform(0.7, 1.3, len(dates))
    dates = rng.choice(dates, size=n, p=w / w.sum())
    products = rng.choice(list(PRODUCT_ISSUES), size=n, p=[0.30, 0.18, 0.22, 0.12, 0.18])
    issues = [rng.choice(PRODUCT_ISSUES[p]) for p in products]
    responses = rng.choice(RESPONSES, size=n, p=RESP_P)
    timely = np.where(rng.random(n) < 0.96, "Yes", "No")
    states = rng.choice(["CA", "TX", "NY", "FL", "IL", "MH", "WA"], size=n)
    return pd.DataFrame({
        "Date received": pd.to_datetime(dates).strftime("%Y-%m-%d"),
        "Product": products, "Issue": issues,
        "Company": rng.choice(COMPANIES, size=n),
        "Company response to consumer": responses,
        "Timely response?": timely, "State": states,
    }).sort_values("Date received").reset_index(drop=True)


def main() -> None:
    ensure_dirs()
    labeled_text().to_csv(DATA_DIR / "demo_financial_text_labeled.csv", index=False)
    unlabeled_text().to_csv(DATA_DIR / "demo_financial_text.csv", index=False)
    complaints().to_csv(DATA_DIR / "demo_complaints.csv", index=False)
    log.info("Wrote synthetic demo files to %s", DATA_DIR)


if __name__ == "__main__":
    main()
