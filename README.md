# FinInsight — NLP-Powered Financial Insights

End-to-end analytics project: a financial-text **sentiment classifier** (Python / scikit-learn) plus a **consumer-complaint analytics** pipeline, packaged into Power BI-ready outputs.

**Stack:** Python · pandas · scikit-learn · matplotlib · pytest · Power BI (DAX)

## Highlights
- **Leak-free ML workflow** — TF-IDF lives inside each sklearn `Pipeline`; stratified train/test split; model selection by cross-validation on the training split only; the test set is used once.
- **Model comparison** — Logistic Regression vs Multinomial Naive Bayes vs calibrated Linear SVM, ranked by CV macro-F1 (mean ± std).
- **Honest evaluation** — accuracy, macro precision/recall/F1, per-class report, confusion matrix, **expected calibration error**, plus an **error-analysis** file of misclassified sentences and **top n-gram features** per class.
- **Robust complaint pipeline** — auto-detects CFPB column names, validates and cleans rows, writes a data-quality report, and builds a star-schema (fact + date dimension + aggregates) so Power BI numbers reconcile.
- **Tested** — `pytest` suite covers loading, cleaning, no-leakage pipeline structure, calibration metric, and aggregate reconciliation.

## Project structure
```
src/
  utils.py               shared paths, logging, data loading (CSV or raw PhraseBank .txt)
  train_model.py         compare models, evaluate, save metrics/plots/model
  score_text.py          score new text -> predictions with confidence + needs_review flag
  complaint_analysis.py  clean + aggregate complaints -> Power BI tables and charts
  make_demo_data.py      generates SYNTHETIC demo data
tests/                   pytest suite
powerbi/POWER_BI_GUIDE.md  relationships, DAX measures, page layout
data/  outputs/  models/
```

## Quick start (Python 3.10+)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make demo        # or run the commands in the Makefile manually
make test
```
Without `make`:
```bash
python src/make_demo_data.py
python src/train_model.py --input data/demo_financial_text_labeled.csv --tag demo
python src/score_text.py --input data/demo_financial_text.csv --model models/sentiment_model_demo.joblib
python src/complaint_analysis.py --input data/demo_complaints.csv
```

> **Demo data is synthetic** (template-generated sentences and random complaints). Pipeline checks on it say nothing about real-world performance. Never report demo metrics on a resume.

## Use real data
**1. Financial PhraseBank** ([dataset page](https://huggingface.co/datasets/takala/financial_phrasebank)) — save as `data/financial_phrasebank.csv` with columns `sentence`, `sentiment` (positive / neutral / negative). The raw `Sentences_*.txt` files (`sentence@label`) also work.
```bash
python src/train_model.py --input data/financial_phrasebank.csv
python src/score_text.py --input data/your_text.csv --text-col text
```
**2. CFPB complaints** ([database](https://www.consumerfinance.gov/data-research/consumer-complaints/)) — export a CSV to `data/cfpb_complaints.csv`.
```bash
python src/complaint_analysis.py --input data/cfpb_complaints.csv
# override detection if needed: --date-col "Date received" --product-col "Product" ...
```
The public CFPB export may not include narratives, so it is used for structured analytics only, not for sentiment training.

## Outputs (`outputs/`)
| File | Purpose |
|---|---|
| `metrics.json`, `classification_report.csv`, `confusion_matrix.csv/.png` | Held-out test evaluation |
| `model_comparison.csv/.png` | CV macro-F1 of each candidate |
| `misclassified_examples.csv`, `top_features.csv` | Error analysis / explainability |
| `sentiment_predictions.csv` | Predictions, class probabilities, `needs_review` flag |
| `complaints_clean.csv`, `dim_date.csv` | Fact + date dimension for Power BI |
| `complaint_monthly_trends.csv` (MoM %, 3-mo avg), `complaints_by_product.csv`, `top_complaint_issues.csv`, `company_response_summary.csv`, `company_scorecard.csv`, `monthly_by_product.csv` | Aggregates |
| `data_quality_report.json` | Rows in/out, bad dates, missing values, date range |
| `chart_*.png` | Static charts for the README |

Power BI setup, relationships and DAX: see [`powerbi/POWER_BI_GUIDE.md`](powerbi/POWER_BI_GUIDE.md).

## Responsible use & limitations
- Sentiment of financial news sentences does not predict stock prices, defaults, fraud or creditworthiness.
- Confidence scores are not guaranteed calibrated; ECE is reported, and low-confidence rows are flagged for human review.
- PhraseBank is financial-news text, not bank-customer complaints — don't claim it models complaint sentiment.
- Remove sensitive personal data and follow dataset terms before publishing.

## Results (fill in after running on real data)
| Item | Value |
|---|---|
| Dataset / size | _e.g. PhraseBank AllAgree, N = …_ |
| Selected model | _…_ |
| Test macro-F1 / accuracy | _…_ |
| Biggest confusion | _e.g. neutral ↔ positive_ |
| CFPB extract date & filters | _…_ |
| Top 3 business findings | _based on your actual data_ |

## Portfolio checklist
- [ ] Train on real Financial PhraseBank data and paste results above
- [ ] Review `misclassified_examples.csv` and write 3–4 lines of error analysis
- [ ] Run the CFPB export; document extraction date and filters
- [ ] Build the Power BI report; add screenshots to `docs/`
- [ ] Add the GitHub link and 2–3 bullet points to your resume
