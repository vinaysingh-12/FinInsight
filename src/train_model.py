"""Train and evaluate a financial-text sentiment classifier.

Pipeline
--------
1. Load + clean labeled data (Financial PhraseBank CSV or raw @-separated txt).
2. Stratified train/test split (test set is untouched until the final evaluation).
3. Compare candidate models with stratified k-fold CV on the TRAIN split only.
4. Refit the CV winner on all training data, evaluate once on the held-out test set.
5. Save metrics, confusion matrix, error analysis, top features and the fitted model.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import (LABELS, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE, ensure_dirs, get_logger,
                   load_labeled_text)

log = get_logger("train_model")


def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           strip_accents="unicode")


def candidate_models() -> dict[str, Pipeline]:
    """TF-IDF lives inside each Pipeline so the vocabulary is fit only on training folds."""
    return {
        "logistic_regression": Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                                       random_state=RANDOM_STATE))]),
        "multinomial_nb": Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", MultinomialNB(alpha=0.3))]),
        "linear_svc_calibrated": Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE), cv=3))]),
    }


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE of the top-class confidence. Lower is better; 0 = perfectly calibrated."""
    conf = proba.max(axis=1)
    correct = (classes[proba.argmax(axis=1)] == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def top_features(pipe: Pipeline, k: int = 15) -> pd.DataFrame | None:
    """Most indicative n-grams per class (linear / NB models only)."""
    clf = pipe.named_steps["clf"]
    names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    if hasattr(clf, "coef_"):
        weights = clf.coef_
    elif hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_ - clf.feature_log_prob_.mean(axis=0)
    else:
        return None
    rows = []
    for i, cls in enumerate(clf.classes_):
        for rank, j in enumerate(np.argsort(weights[i])[::-1][:k], start=1):
            rows.append({"class": cls, "rank": rank, "feature": names[j],
                         "weight": round(float(weights[i][j]), 4)})
    return pd.DataFrame(rows)


def plot_confusion(cm: np.ndarray, labels: list[str], path) -> None:
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (held-out test set)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised share")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison(cv_df: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(cv_df["model"], cv_df["cv_macro_f1_mean"], xerr=cv_df["cv_macro_f1_std"],
            color="#2a6fb0", capsize=3)
    ax.set_xlabel("Macro F1 (cross-validation on training split)")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True, help="Labeled CSV (sentence,sentiment) or PhraseBank .txt")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--tag", default="", help="Suffix for output files (e.g. 'demo')")
    args = ap.parse_args()

    ensure_dirs()
    sfx = f"_{args.tag}" if args.tag else ""
    df = load_labeled_text(args.input)
    log.info("Loaded %d labeled sentences | class counts: %s", len(df),
             df["sentiment"].value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        df["sentence"], df["sentiment"], test_size=args.test_size,
        stratify=df["sentiment"], random_state=RANDOM_STATE)

    # ---- model selection: CV on the training split only -----------------------------
    n_splits = max(2, min(args.cv_folds, int(y_train.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    models = candidate_models()
    cv_rows = []
    for name, pipe in models.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, n_jobs=-1,
                             scoring={"acc": "accuracy", "f1": "f1_macro"})
        cv_rows.append({"model": name,
                        "cv_accuracy_mean": res["test_acc"].mean(), "cv_accuracy_std": res["test_acc"].std(),
                        "cv_macro_f1_mean": res["test_f1"].mean(), "cv_macro_f1_std": res["test_f1"].std()})
        log.info("%-24s CV macro-F1 %.3f ± %.3f", name, res["test_f1"].mean(), res["test_f1"].std())
    cv_df = pd.DataFrame(cv_rows).sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    best_name = cv_df.loc[0, "model"]
    log.info("Selected model (by CV macro-F1): %s", best_name)

    # ---- final fit + one-shot test evaluation ---------------------------------------
    best = models[best_name].fit(X_train, y_train)
    classes = best.classes_
    proba = best.predict_proba(X_test)
    y_pred = classes[proba.argmax(axis=1)]

    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=LABELS,
                                                  average="macro", zero_division=0)
    ece = expected_calibration_error(y_test.to_numpy(), proba, classes)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    report = classification_report(y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T.loc[LABELS + ["macro avg", "weighted avg"]].round(4)
    per_class.index.name = "class"

    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file": str(args.input), "selected_model": best_name,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)), "cv_folds": n_splits,
        "test_accuracy": round(acc, 4), "test_macro_precision": round(p, 4),
        "test_macro_recall": round(r, 4), "test_macro_f1": round(f1, 4),
        "expected_calibration_error": round(ece, 4),
        "note": "Test metrics are from a single held-out split. Synthetic demo data is not real-world performance.",
    }

    # ---- artifacts ------------------------------------------------------------------
    cv_df.round(4).to_csv(OUTPUT_DIR / f"model_comparison{sfx}.csv", index=False)
    per_class.to_csv(OUTPUT_DIR / f"classification_report{sfx}.csv")
    pd.DataFrame(cm, index=[f"actual_{l}" for l in LABELS],
                 columns=[f"pred_{l}" for l in LABELS]).to_csv(OUTPUT_DIR / f"confusion_matrix{sfx}.csv")
    (OUTPUT_DIR / f"metrics{sfx}.json").write_text(json.dumps(metrics, indent=2))

    test_out = pd.DataFrame({"sentence": X_test.values, "actual": y_test.values, "predicted": y_pred,
                             "confidence": proba.max(axis=1).round(4)})
    test_out["correct"] = test_out["actual"] == test_out["predicted"]
    test_out.to_csv(OUTPUT_DIR / f"test_predictions{sfx}.csv", index=False)
    (test_out[~test_out["correct"]].sort_values("confidence", ascending=False)
     .to_csv(OUTPUT_DIR / f"misclassified_examples{sfx}.csv", index=False))

    feats = top_features(best)
    if feats is not None:
        feats.to_csv(OUTPUT_DIR / f"top_features{sfx}.csv", index=False)

    plot_confusion(cm, LABELS, OUTPUT_DIR / f"confusion_matrix{sfx}.png")
    plot_comparison(cv_df, OUTPUT_DIR / f"model_comparison{sfx}.png")

    joblib.dump({"pipeline": best, "labels": list(classes), "model_name": best_name,
                 "trained_at_utc": metrics["trained_at_utc"], "metrics": metrics},
                MODEL_DIR / f"sentiment_model{sfx}.joblib")

    log.info("TEST  accuracy=%.3f  macro-P=%.3f  macro-R=%.3f  macro-F1=%.3f  ECE=%.3f",
             acc, p, r, f1, ece)
    log.info("Misclassified: %d / %d -> outputs/misclassified_examples%s.csv",
             int((~test_out['correct']).sum()), len(test_out), sfx)


if __name__ == "__main__":
    main()
