"""End-to-end smoke run using committed sample datasets.

This validates:
  - dataset contracts
  - TabFM classification + regression paths
  - TimesFM forecast + backtest paths
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from app_core.tabfm_service import run_tabfm_prediction
from app_core.timesfm_service import run_timesfm_backtest, run_timesfm_forecast


ROOT = Path(__file__).resolve().parents[1]


def _set_cache_env_defaults() -> None:
    os.environ.setdefault("UV_CACHE_DIR", str(ROOT / ".uv-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / ".cache" / "hf" / "hub"))
    # Ensure smoke tests don't depend on live network once caches are present.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> int:
    _set_cache_env_defaults()

    out_dir = ROOT / "artifacts" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {}

    # TabFM: classification
    print("[smoke] TabFM classification: iris", flush=True)
    iris_train = _load_csv(ROOT / "data" / "samples" / "tabular" / "iris" / "train.csv")
    iris_predict = _load_csv(ROOT / "data" / "samples" / "tabular" / "iris" / "predict.csv")
    iris_out = run_tabfm_prediction(
        train_df=iris_train,
        predict_df=iris_predict,
        target_column="target",
        task_mode="classification",
    )
    assert not iris_out.output_df.empty
    (out_dir / "tabfm_iris_predictions.csv").write_text(
        iris_out.output_df.to_csv(index=False), encoding="utf-8"
    )
    results["tabfm_iris"] = {"task": iris_out.task, "rows": len(iris_out.output_df)}

    # TabFM: regression
    print("[smoke] TabFM regression: wine_quality_red", flush=True)
    wine_train = _load_csv(
        ROOT / "data" / "samples" / "tabular" / "wine_quality_red" / "train.csv"
    )
    wine_predict = _load_csv(
        ROOT / "data" / "samples" / "tabular" / "wine_quality_red" / "predict.csv"
    )
    wine_out = run_tabfm_prediction(
        train_df=wine_train,
        predict_df=wine_predict,
        target_column="target",
        task_mode="regression",
    )
    assert not wine_out.output_df.empty
    (out_dir / "tabfm_wine_predictions.csv").write_text(
        wine_out.output_df.to_csv(index=False), encoding="utf-8"
    )
    results["tabfm_wine"] = {"task": wine_out.task, "rows": len(wine_out.output_df)}

    # TimesFM: forecast + backtest
    print("[smoke] TimesFM forecast/backtest: air_passengers", flush=True)
    air = _load_csv(
        ROOT / "data" / "samples" / "timeseries" / "air_passengers" / "history.csv"
    )
    forecast = run_timesfm_forecast(
        history_df=air,
        horizon=24,
        max_context=512,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=True,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        use_xreg=False,
    )
    assert not forecast.forecast_df.empty
    (out_dir / "timesfm_air_forecast.csv").write_text(
        forecast.forecast_df.to_csv(index=False), encoding="utf-8"
    )

    backtest = run_timesfm_backtest(
        history_df=air,
        holdout_points=12,
        max_context=512,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=True,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        use_xreg=False,
    )
    assert not backtest.comparison_df.empty
    (out_dir / "timesfm_air_backtest.csv").write_text(
        backtest.comparison_df.to_csv(index=False), encoding="utf-8"
    )
    results["timesfm_air"] = {
        "forecast_rows": len(forecast.forecast_df),
        "backtest_rows": len(backtest.comparison_df),
        "metrics": backtest.metrics,
    }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
