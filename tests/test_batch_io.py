"""Tests for batch CSV ingestion utilities."""

from __future__ import annotations

import io
import zipfile

import pytest

from app_core.batch_io import load_batch_items_from_bytes
from app_core.validators import ValidationError


def _make_zip(entries: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return payload.getvalue()


def test_load_batch_items_from_csv_and_zip_dedupes_filenames() -> None:
    zip_bytes = _make_zip(
        {
            "nested/predict.csv": b"age\n30\n",
            "predict_extra.csv": b"age\n31\n",
        }
    )
    items = load_batch_items_from_bytes(
        csv_files=[("predict.csv", b"age\n29\n")],
        zip_files=[("batch.zip", zip_bytes)],
        max_files=25,
    )

    assert [item.name for item in items] == ["predict.csv", "predict_2.csv", "predict_extra.csv"]


def test_load_batch_items_rejects_non_csv_entries_in_zip() -> None:
    zip_bytes = _make_zip({"notes.txt": b"hello"})
    with pytest.raises(ValidationError, match="Non-CSV"):
        load_batch_items_from_bytes(
            csv_files=[],
            zip_files=[("bad.zip", zip_bytes)],
            max_files=25,
        )


def test_load_batch_items_enforces_max_files() -> None:
    csv_files = [(f"f_{idx}.csv", b"age\n1\n") for idx in range(26)]
    with pytest.raises(ValidationError, match="Maximum batch size"):
        load_batch_items_from_bytes(
            csv_files=csv_files,
            zip_files=[],
            max_files=25,
        )

