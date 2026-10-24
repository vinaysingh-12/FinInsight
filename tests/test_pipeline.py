import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import complaint_analysis as ca  # noqa: E402
import make_demo_data as demo  # noqa: E402
import train_model as tm  # noqa: E402
from utils import load_labeled_text  # noqa: E402


def test_load_labeled_text_cleans_and_dedupes(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"Sentence": ["a good quarter", "a good quarter", "bad news", "meh", "x"],
                  "Sentiment": ["Positive", "positive", "negative", "unknown", ""]}).to_csv(p, index=False)
    df = load_labeled_text(p)
    assert list(df.columns) == ["sentence", "sentiment"]
    assert len(df) == 2  # duplicate + invalid labels removed


def test_load_phrasebank_txt(tmp_path):
    p = tmp_path / "Sentences_AllAgree.txt"
    p.write_text("Profit rose 10% .@positive\nSales fell .@negative\nbad line without label\n", encoding="latin-1")
    df = load_labeled_text(p)
    assert df["sentiment"].tolist() == ["positive", "negative"]


def test_ece_perfect_and_worst():
    classes = np.array(["a", "b"])
    y = np.array(["a", "b"])
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    wrong_confident = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert tm.expected_calibration_error(y, perfect, classes) == pytest.approx(0.0)
    assert tm.expected_calibration_error(y, wrong_confident, classes) == pytest.approx(1.0)


def test_pipelines_keep_tfidf_inside():
    for pipe in tm.candidate_models().values():
        assert list(pipe.named_steps)[0] == "tfidf"  # no vocabulary leakage across folds


def test_model_trains_and_predicts_proba():
    df = demo.labeled_text(60)
    pipe = tm.candidate_models()["logistic_regression"].fit(df["sentence"], df["sentiment"])
    proba = pipe.predict_proba(["Profit increased strongly"])
    assert proba.shape == (1, 3) and proba.sum() == pytest.approx(1.0)


def test_detect_columns_and_clean():
    raw = demo.complaints(300)
    m = ca.detect_columns(raw, {})
    assert m["date"] == "Date received" and m["timely"] == "Timely response?"
    clean, qa = ca.clean(raw, m)
    assert qa["rows_out"] == len(clean) == 300
    assert {"year_month", "got_relief", "timely_flag"} <= set(clean.columns)


def test_clean_drops_bad_rows():
    raw = demo.complaints(50)
    raw.loc[0, "Date received"] = "not a date"
    raw.loc[1, "Product"] = ""
    clean, qa = ca.clean(raw, ca.detect_columns(raw, {}))
    assert qa["rows_dropped"] == 2 and qa["unparseable_dates"] == 1


def test_aggregates_reconcile_to_total():
    raw = demo.complaints(500)
    clean, _ = ca.clean(raw, ca.detect_columns(raw, {}))
    t = ca.aggregate(clean, top_n=100)
    assert t["complaint_monthly_trends"]["complaint_count"].sum() == len(clean)
    assert t["complaints_by_product"]["complaint_count"].sum() == len(clean)
    assert t["company_response_summary"]["complaint_count"].sum() == len(clean)


def test_missing_required_column_raises():
    with pytest.raises(SystemExit):
        ca.detect_columns(pd.DataFrame({"foo": [1]}), {})
