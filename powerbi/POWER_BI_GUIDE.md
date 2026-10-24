# Power BI build guide

## 1. Load tables (Get Data → Text/CSV)
| Table | File | Role |
|---|---|---|
| `complaints` | `outputs/complaints_clean.csv` | **Fact** (one row per complaint) |
| `dim_date` | `outputs/dim_date.csv` | **Date dimension** |
| `company_scorecard` | `outputs/company_scorecard.csv` | Reference table |
| `sentiment` | `outputs/sentiment_predictions.csv` | Separate page (text sentiment) |
| `model_comparison`, `classification_report`, `confusion_matrix` | `outputs/*.csv` | Model Quality page |

## 2. Relationships
`dim_date[date]` (1) → (*) `complaints[date_received]` — mark `dim_date` as the date table.
Calculate everything from `complaints`; do **not** add up the pre-aggregated CSVs (that double counts).

## 3. DAX measures
```dax
Total Complaints = COUNTROWS(complaints)

Complaints Last Month =
CALCULATE([Total Complaints], DATEADD(dim_date[date], -1, MONTH))

MoM Change % =
DIVIDE([Total Complaints] - [Complaints Last Month], [Complaints Last Month])

Relief Rate % = DIVIDE(SUM(complaints[got_relief]), [Total Complaints])

Timely Response % = DIVIDE(SUM(complaints[timely_flag]), [Total Complaints])

3M Rolling Avg =
AVERAGEX(DATESINPERIOD(dim_date[date], MAX(dim_date[date]), -3, MONTH),
         CALCULATE([Total Complaints]))   -- use month-level axis (dim_date[year_month])

Share of Total % =
DIVIDE([Total Complaints], CALCULATE([Total Complaints], ALL(complaints[product])))
```
Sentiment page:
```dax
Texts Scored = COUNTROWS(sentiment)
Negative Share % = DIVIDE(CALCULATE([Texts Scored], sentiment[predicted_sentiment] = "negative"), [Texts Scored])
Needs Review = CALCULATE([Texts Scored], sentiment[needs_review] = TRUE())
```

## 4. Pages
1. **Executive Overview** — KPI cards (Total, MoM %, Relief %, Timely %), monthly line + 3M avg, product donut/bar, top-10 issues.
2. **Complaint Deep Dive** — slicers: year, product, company; matrix product × issue; response-type stacked bar; state map if available.
3. **Financial Text Sentiment** — sentiment distribution, avg confidence, table of `needs_review` rows.
4. **Model Quality** — model comparison bar, per-class precision/recall/F1, confusion matrix; label as *held-out test set*.

## 5. Design tips
One accent colour + greys; titles that state the takeaway ("Credit card complaints drive ~1/3 of volume"); sort bars descending; add a text box with data source and extraction date.
