"""Warm TimesFM + TabFM model weights and caches.

This script performs tiny real inference runs to force:
  - pip/uv dependencies to import correctly
  - model weights to download and cache
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from app_core.timesfm_service import run_timesfm_forecast
from app_core.tabfm_service import run_tabfm_prediction


def _set_cache_env_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("UV_CACHE_DIR", str(root / ".uv-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / ".cache" / "hf" / "hub"))
    # Increase Hub timeouts: TabFM weights can be large and slow on some networks.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


def _warm_timesfm() -> None:
    ts = pd.date_range("2020-01-01", periods=128, freq="D")
    values = np.sin(np.linspace(0, 10, len(ts))).astype(float)
    history_df = pd.DataFrame({"timestamp": ts, "value": values})
    _ = run_timesfm_forecast(
        history_df=history_df,
        horizon=16,
        max_context=256,
        max_horizon=256,
        normalize_inputs=True,
        include_quantiles=True,
        model_id="google/timesfm-2.5-200m-pytorch",
        backend="torch",
        use_xreg=False,
    )


def _warm_tabfm() -> None:
    # Pre-download checkpoints with conservative settings (single worker) to avoid
    # flaky parallel downloads, then run a tiny inference.
    from huggingface_hub import snapshot_download

    cache_dir = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "google/tabfm-1.0.0-pytorch"
    for subfolder in ("classification", "regression"):
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/**"],
            cache_dir=cache_dir,
            max_workers=1,
            resume_download=True,
        )

    rng = np.random.default_rng(42)
    x1 = rng.normal(size=80)
    x2 = rng.normal(size=80)
    y = (x1 + 0.3 * x2 > 0).astype(int)
    train_df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    predict_df = pd.DataFrame({"x1": rng.normal(size=10), "x2": rng.normal(size=10)})
    _ = run_tabfm_prediction(
        train_df=train_df,
        predict_df=predict_df,
        target_column="target",
        task_mode="classification",
    )


def main() -> int:
    _set_cache_env_defaults()

    out_dir = Path("artifacts") / "warm"
    out_dir.mkdir(parents=True, exist_ok=True)

    status: dict[str, object] = {"timesfm": "pending", "tabfm": "pending"}

    try:
        _warm_timesfm()
        status["timesfm"] = "ok"
    except Exception as exc:
        status["timesfm"] = f"failed: {exc}"

    try:
        _warm_tabfm()
        status["tabfm"] = "ok"
    except Exception as exc:
        status["tabfm"] = f"failed: {exc}"

    (out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if all(v == "ok" for v in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
