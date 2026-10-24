"""Shared helpers for FinInsight: paths, logging and data loading."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

LABELS = ["negative", "neutral", "positive"]
RANDOM_STATE = 42


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger(name)


def ensure_dirs() -> None:
    for d in (DATA_DIR, OUTPUT_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_labeled_text(path: str | Path) -> pd.DataFrame:
    """Load labeled sentiment data.

    Supports:
      * CSV with columns `sentence` and `sentiment`
      * Financial PhraseBank raw text files (`Sentences_*.txt`), one
        `sentence@label` per line.
    Returns a clean DataFrame with columns `sentence`, `sentiment`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See README -> 'Train the model with real labeled data'.")
    if path.suffix.lower() == ".txt":
        rows = []
        for line in path.read_text(encoding="latin-1").splitlines():
            if "@" not in line:
                continue
            sentence, label = line.rsplit("@", 1)
            rows.append((sentence.strip(), label.strip()))
        df = pd.DataFrame(rows, columns=["sentence", "sentiment"])
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"sentence", "sentiment"} - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df = df[["sentence", "sentiment"]].copy()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[df["sentiment"].isin(LABELS) & (df["sentence"].str.len() > 0)]
    before = len(df)
    df = df.drop_duplicates(subset="sentence").reset_index(drop=True)
    if len(df) < before:
        get_logger("utils").info("Dropped %d duplicate sentences", before - len(df))
    return df
