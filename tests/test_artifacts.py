"""Tests for artifact bundle generation."""

from __future__ import annotations

import io
import zipfile

from app_core.artifacts import build_artifact_bundle


def test_build_artifact_bundle_contains_manifest_and_files() -> None:
    bundle = build_artifact_bundle(
        files={
            "tabfm_predictions.csv": b"a,b\n1,2\n",
            "timesfm_forecast.csv": b"t,pred\n2024-01-01,10\n",
        },
        metadata={"project": "timesfm-tabfm-app", "run_id": "abc-123"},
    )

    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert "tabfm_predictions.csv" in names
        assert "timesfm_forecast.csv" in names

