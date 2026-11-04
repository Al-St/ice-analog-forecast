"""C08 (review round 3): the seasonal/interannual terminology mapping.

The manuscript calls the 12/24/36-month scheme the "interannual scale";
the CLI historically spells it `seasonal`.  The alias must resolve to
the canonical value so a run's metadata and outputs are identical under
either spelling (stored SHAs keep validating).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_forecast.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_forecast", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_forecast"] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_scale_identity() -> None:
    rf = _load_script()
    assert rf.canonical_scale("1") == "1"
    assert rf.canonical_scale("seasonal") == "seasonal"


def test_interannual_alias_resolves_to_seasonal() -> None:
    rf = _load_script()
    assert rf.canonical_scale("interannual") == "seasonal"


def test_alias_is_declared_in_choices() -> None:
    rf = _load_script()
    assert "seasonal" in rf.SCALE_CHOICES
    assert "interannual" in rf.SCALE_CHOICES
    assert set(rf.SCALE_ALIASES) == {"interannual"}
