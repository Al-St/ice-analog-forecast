"""Shared fixtures: src path and synthetic data builders."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from analog_ice import MonthCalendar  # noqa: E402


@pytest.fixture()
def calendar() -> MonthCalendar:
    """A 45-year synthetic record starting 1979-01 (540 months)."""
    return MonthCalendar.make(1979, 540)


@pytest.fixture()
def synthetic_sic(calendar) -> np.ndarray:
    """(8, 6, 540) float64 concentrations in [0, 1] with a seasonal cycle
    and a mild linear trend; deterministic (seeded generator)."""
    rng = np.random.default_rng(42)
    h, w, n = 8, 6, calendar.n_months
    seasonal = 0.5 + 0.3 * np.cos(2.0 * np.pi * (calendar.months - 3) / 12.0)
    trend = np.arange(n) / (12.0 * 540.0) * 0.1
    field = (seasonal[None, None, :] + trend[None, None, :]
             + 0.05 * rng.standard_normal((h, w, n)))
    return np.clip(field, 0.0, 1.0)
