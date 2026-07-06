"""Download and normalize example datasets for TabFM and TimesFM.

Writes:
  - data/raw/...     (full downloads, gitignored)
  - data/samples/... (tiny samples, committed)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLES = ROOT / "data" / "samples"
DATA_MANIFEST = ROOT / "data" / "manifest.json"


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str
    kind: str  # tabular|timeseries
    notes: str


SOURCES: list[DatasetSource] = [
    DatasetSource(
        name="iris",
        kind="tabular",
        url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        notes="Seaborn iris dataset (species classification).",
    ),
    DatasetSource(
        name="adult_income",
        kind="tabular",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        notes="UCI Adult income (binary classification). No header row.",
    ),
    DatasetSource(
        name="wine_quality_red",
        kind="tabular",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        notes="UCI Wine Quality (regression target=quality). CSV is semicolon-delimited.",
    ),
    DatasetSource(
        name="air_passengers",
        kind="timeseries",
        url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
        notes="AirPassengers monthly totals (Month,Passengers).",
    ),
    DatasetSource(
        name="daily_min_temperatures",
        kind="timeseries",
        url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        notes="Daily minimum temperatures (Date,Temp).",
    ),
    DatasetSource(
        name="monthly_sunspots",
        kind="timeseries",
        url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv",
        notes="Monthly sunspots (Month,Sunspots).",
    ),
]


def _http_get(url: str, timeout_seconds: int = 60) -> bytes:
    resp = requests.get(url, timeout=timeout_seconds)
    resp.raise_for_status()
    return resp.content


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _normalize_tabular_iris(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    # Standardize target column name.
    if "species" in df.columns:
        df = df.rename(columns={"species": "target"})
    return df


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "target",
]


def _normalize_tabular_adult_income(raw_path: Path) -> pd.DataFrame:
    # adult.data is comma-separated with possible whitespace; also contains a trailing period
    # only in adult.test (we don't use test here).
    rows: list[list[str]] = []
    with raw_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            cleaned = [cell.strip() for cell in row if cell is not None]
            if len(cleaned) != len(ADULT_COLUMNS):
                # Skip malformed lines.
                continue
            rows.append(cleaned)
    df = pd.DataFrame(rows, columns=ADULT_COLUMNS)
    # Normalize target values.
    df["target"] = df["target"].astype(str).str.replace(".", "", regex=False).str.strip()
    return df


def _normalize_tabular_wine_quality(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, sep=";")
    if "quality" in df.columns:
        df = df.rename(columns={"quality": "target"})
    return df


def _train_predict_split(
    df: pd.DataFrame,
    target_col: str = "target",
    train_rows: int = 120,
    predict_rows: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    if target_col not in df.columns:
        raise ValueError(f"Expected '{target_col}' in dataframe.")

    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(df) < train_rows + predict_rows:
        # If small dataset, just take as many as possible while keeping both splits non-empty.
        train_rows = max(10, int(len(df) * 0.8))
        predict_rows = max(5, len(df) - train_rows)

    idx = rng.permutation(len(df))
    train_idx = idx[:train_rows]
    predict_idx = idx[train_rows : train_rows + predict_rows]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    predict_df = df.iloc[predict_idx].drop(columns=[target_col]).reset_index(drop=True)
    return train_df, predict_df


def _normalize_timeseries_generic(
    raw_path: Path,
    timestamp_candidates: list[str],
    value_candidates: list[str],
) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    timestamp_col = next((c for c in timestamp_candidates if c in df.columns), None)
    value_col = next((c for c in value_candidates if c in df.columns), None)
    if timestamp_col is None or value_col is None:
        raise ValueError(
            f"Could not find timestamp/value columns in {raw_path.name}. "
            f"Columns={list(df.columns)}"
        )

    out = df[[timestamp_col, value_col]].copy()
    out = out.rename(columns={timestamp_col: "timestamp", value_col: "value"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)
    return out


def _write_tabular_sample(dataset_name: str, normalized: pd.DataFrame) -> None:
    train_df, predict_df = _train_predict_split(normalized, target_col="target")
    out_dir = DATA_SAMPLES / "tabular" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    predict_df.to_csv(out_dir / "predict.csv", index=False)


def _write_timeseries_sample(dataset_name: str, normalized: pd.DataFrame, sample_points: int = 240) -> None:
    out_dir = DATA_SAMPLES / "timeseries" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = normalized.tail(sample_points).reset_index(drop=True)
    sample.to_csv(out_dir / "history.csv", index=False)


def _fallback_synthetic() -> None:
    # Tabular classification
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    target = (x1 + 0.5 * x2 > 0).astype(int)
    df_cls = pd.DataFrame({"x1": x1, "x2": x2, "target": target})
    _write_tabular_sample("synthetic_classification", df_cls)

    # Tabular regression
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    y = 3.0 * x1 - 2.0 * x2 + rng.normal(scale=0.2, size=200)
    df_reg = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    _write_tabular_sample("synthetic_regression", df_reg)

    # Time-series
    ts = pd.date_range("2020-01-01", periods=400, freq="D")
    values = np.sin(np.linspace(0, 20, len(ts))) + rng.normal(scale=0.1, size=len(ts))
    df_ts = pd.DataFrame({"timestamp": ts, "value": values})
    _write_timeseries_sample("synthetic_sine", df_ts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabular", action="store_true", help="Download/prepare tabular datasets.")
    parser.add_argument("--timeseries", action="store_true", help="Download/prepare time-series datasets.")
    parser.add_argument("--all", action="store_true", help="Download/prepare all datasets.")
    parser.add_argument("--force", action="store_true", help="Re-download even if raw file exists.")
    parser.add_argument("--allow-synthetic-fallback", action="store_true", default=True)
    args = parser.parse_args()

    want_tabular = args.all or args.tabular or (not args.timeseries and not args.tabular and not args.all)
    want_timeseries = args.all or args.timeseries or (not args.timeseries and not args.tabular and not args.all)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    (DATA_SAMPLES / "tabular").mkdir(parents=True, exist_ok=True)
    (DATA_SAMPLES / "timeseries").mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"sources": [s.__dict__ for s in SOURCES]}
    _write_json(DATA_MANIFEST, manifest)

    any_failed = False

    for src in SOURCES:
        if src.kind == "tabular" and not want_tabular:
            continue
        if src.kind == "timeseries" and not want_timeseries:
            continue

        raw_path = DATA_RAW / src.kind / f"{src.name}.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if args.force or not raw_path.exists():
                content = _http_get(src.url)
                _write_bytes(raw_path, content)

            if src.kind == "tabular":
                if src.name == "iris":
                    normalized = _normalize_tabular_iris(raw_path)
                elif src.name == "adult_income":
                    normalized = _normalize_tabular_adult_income(raw_path)
                elif src.name == "wine_quality_red":
                    normalized = _normalize_tabular_wine_quality(raw_path)
                else:
                    raise ValueError(f"Unknown tabular dataset: {src.name}")
                _write_tabular_sample(src.name, normalized)
            else:
                if src.name == "air_passengers":
                    normalized = _normalize_timeseries_generic(
                        raw_path, timestamp_candidates=["Month", "month", "date", "Date"], value_candidates=["Passengers", "passengers", "value"]
                    )
                elif src.name == "daily_min_temperatures":
                    normalized = _normalize_timeseries_generic(
                        raw_path, timestamp_candidates=["Date", "date"], value_candidates=["Temp", "temp", "Temperature", "temperature", "value"]
                    )
                elif src.name == "monthly_sunspots":
                    normalized = _normalize_timeseries_generic(
                        raw_path, timestamp_candidates=["Month", "month", "Date", "date"], value_candidates=["Sunspots", "sunspots", "value"]
                    )
                else:
                    raise ValueError(f"Unknown timeseries dataset: {src.name}")
                _write_timeseries_sample(src.name, normalized)

        except Exception as exc:
            any_failed = True
            print(f"[WARN] Failed dataset '{src.name}': {exc}")

    if any_failed and args.allow_synthetic_fallback:
        print("[WARN] Some downloads failed; generating synthetic fallback samples.")
        _fallback_synthetic()

    print("OK: datasets prepared under data/samples/ (and full downloads under data/raw/).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
