"""Monthly calendar for a contiguous 1979-01.. record.

A tiny replacement for datetime/pandas machinery: the model only needs
calendar months, years, and index arithmetic on a contiguous monthly grid
starting at a fixed year/month. All indices are 0-based positions in the
record; "calendar month" (``cal``) is 1..12.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonthCalendar:
    """Contiguous monthly grid ``start_year-01`` .. ``start_year+N-1-12``.

    Attributes:
        start_year: first year of the record (January).
        n_months: total number of months.
        months: (n,) int array of calendar months 1..12.
        years: (n,) int array of calendar years.
    """

    start_year: int
    n_months: int
    months: np.ndarray
    years: np.ndarray

    @staticmethod
    def make(start_year: int, n_months: int) -> "MonthCalendar":
        """Build a calendar for ``n_months`` consecutive months from January
        of ``start_year``."""
        pos = np.arange(n_months, dtype=np.int64)
        months = (pos % 12) + 1
        years = start_year + pos // 12
        return MonthCalendar(start_year, n_months, months, years)

    def year_month(self, idx: int) -> tuple[int, int]:
        """(year, month) of record position ``idx``."""
        return int(self.years[idx]), int(self.months[idx])

    def index_of(self, year: int, month: int) -> int:
        """Record position of ``year-month`` (the grid is contiguous, so the
        mapping is closed-form)."""
        if not 1 <= month <= 12:
            raise ValueError(f"calendar month out of range: {month}")
        idx = (year - self.start_year) * 12 + (month - 1)
        if not 0 <= idx < self.n_months:
            raise ValueError(f"(year, month) outside the record: {year}-{month:02d}")
        return int(idx)

    def label(self, idx: int) -> str:
        """``YYYY-MM`` label of record position ``idx``."""
        y, m = self.year_month(idx)
        return f"{y:04d}-{m:02d}"
