"""Serialization of a forecast run: the .npz prediction file and its
metadata sidecar.

The npz layout is (key order matters for byte-identical files):
``preds``, ``truths``, ``target_ym``, ``issue_ym``, ``lead_months``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .forecast import ForecastResult


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file (hex)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload(result: ForecastResult) -> dict[str, np.ndarray]:
    """The npz payload arrays of a forecast result, in the canonical key
    order."""
    return {
        "preds": result.preds,
        "truths": result.truths,
        "target_ym": result.target_labels,
        "issue_ym": result.issue_labels,
        "lead_months": np.asarray(result.lead_months, dtype=np.int32),
    }


def save_run(result: ForecastResult,
             out_dir: Path,
             metadata: Optional[dict] = None) -> tuple[Path, str]:
    """Write ``predictions.npz`` (+ SHA-256) and ``metadata.json`` into
    ``out_dir``; return (npz path, sha256 hex)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "predictions.npz"
    np.savez_compressed(npz_path, **payload(result))
    digest = sha256_file(npz_path)
    meta = dict(metadata or {})
    meta["npz_sha256"] = digest
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return npz_path, digest


def load_run(npz_path: Path) -> dict[str, np.ndarray]:
    """Load a saved predictions.npz back into the payload dict."""
    with np.load(Path(npz_path)) as archive:
        return {key: archive[key] for key in
                ("preds", "truths", "target_ym", "issue_ym", "lead_months")}
