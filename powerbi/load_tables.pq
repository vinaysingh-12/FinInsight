// Power BI: Home > Transform data > New Source > Blank Query > Advanced Editor.
// Paste ONE block per query, change `Root` to the folder holding your outputs, then name the query as shown.

// ---- Query: complaints ----
let
    Root = "C:\path\to\fininsight\outputs\",
    Source = Csv.Document(File.Contents(Root & "complaints_clean.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"date_received", type date}, {"year_month", type date},
        {"got_relief", Int64.Type}, {"timely_flag", Int64.Type}})
in Typed

// ---- Query: dim_date ----
let
    Root = "C:\path\to\fininsight\outputs\",
    Source = Csv.Document(File.Contents(Root & "dim_date.csv"), [Delimiter=",", Encoding=65001]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"date", type date}, {"year", Int64.Type}, {"month_num", Int64.Type},
        {"year_month", type date}, {"week_start", type date}, {"is_weekend", Int64.Type}})
in Typed

// ---- Query: sentiment ----
let
    Root = "C:\path\to\fininsight\outputs\",
    Source = Csv.Document(File.Contents(Root & "sentiment_predictions.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {{"confidence", type number}, {"prob_negative", type number},
        {"prob_neutral", type number}, {"prob_positive", type number}, {"needs_review", type logical}})
in Typed

// Model Quality page: load model_comparison.csv, classification_report.csv, confusion_matrix.csv the same way
// (Csv.Document + Table.PromoteHeaders). If you trained with --tag demo, the files end in _demo.
