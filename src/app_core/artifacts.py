"""Artifact bundle helpers."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from typing import Mapping


def build_artifact_bundle(files: Mapping[str, bytes], metadata: Mapping[str, object]) -> bytes:
    """Build an in-memory ZIP bundle from generated artifacts."""
    payload = io.BytesIO()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": sorted(list(files.keys())),
        "metadata": dict(metadata),
    }

    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        for filename, content in files.items():
            archive.writestr(filename, content)

    return payload.getvalue()

