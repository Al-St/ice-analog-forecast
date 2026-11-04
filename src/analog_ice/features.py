"""Causal per-boundary feature frames.

A "frame" is the feature matrix of a case whose issue month is the
boundary ``b``: rows are the record positions ``t <= b`` (the case window
includes its own issue month), columns are the predictor series.

Predictor modes (``mode``):

- ``raw``: the series as-is (the minimal scalar arm);
- ``sigma_only``: divided by the trailing same-calendar-month std (level kept);
- ``m3``: causal anomalization — subtract the trailing-30 same-calendar-month
  mean and divide by its std (ddof = 0). Calendar months with fewer than
  ``min_points`` occurrences strictly before ``b`` are invalid; invalid rows
  are zeroed.

Decomposition (``pca``):

- ``identity``: keep the first series column as-is (no centering);
- ``pca5``: deterministic PCA to 5 components fitted CAUSALLY on the valid
  prefix rows ``t <= b`` (full SVD; each component's sign is fixed so its
  largest-|score| entry is positive, matching the deterministic reference
  decomposition), then applied to every prefix row.

Frames are cached per boundary: a month used both as a query issue and as
a candidate shares one computation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import linalg

PCA_DIM = 5
M3_WINDOW = 30
MIN_POINTS = 10
MIN_FIT_ROWS = 10


@dataclass
class PcaModel:
    """A fitted PCA projection (deterministic full SVD)."""

    mean: np.ndarray        # (p,) mean of the fitted rows
    components: np.ndarray  # (k, p) orthonormal components

    def project(self, matrix: np.ndarray) -> np.ndarray:
        """Project (n, p) rows onto the components -> (n, k)."""
        return (matrix - self.mean) @ self.components.T


def fit_pca(fit_rows: np.ndarray, n_components: int = PCA_DIM) -> PcaModel:
    """Fit PCA on ``fit_rows`` (n, p), full SVD of the centered matrix.

    Components are ordered by descending singular value; each component's
    sign is fixed so its largest-|score| entry over the fitted rows is
    positive (a deterministic convention, applied identically at every
    boundary).

    Raises ``ValueError`` when fewer than ``n_components`` rows or columns
    are available (the decomposition would be underdetermined).
    """
    if fit_rows.shape[0] < n_components or fit_rows.shape[1] < n_components:
        raise ValueError(
            f"PCA-{n_components} needs >= {n_components} rows and columns, "
            f"got {fit_rows.shape}")
    mean = fit_rows.mean(axis=0)
    centered = fit_rows - mean
    _, _, vt = linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].copy()
    fitted = components @ centered.T            # (k, n_fit)
    max_abs = np.argmax(np.abs(fitted), axis=1)  # (k,)
    signs = np.sign(fitted[np.arange(len(max_abs)), max_abs])
    signs[signs == 0] = 1.0
    components *= signs[:, None]
    return PcaModel(mean=mean, components=components)


@dataclass
class Frame:
    """Feature frame of one boundary month.

    Attributes:
        boundary: record position of the case's issue month.
        scores: (boundary + 1, n_series) feature rows, float64.
        n_valid_rows: prefix rows with a valid trailing climatology.
        fit_ok: whether the frame can serve as a case (``raw`` frames are
            always usable; anomalized frames need >= 10 valid rows).
        n_zeroed_rows: prefix rows zeroed by the ``min_points`` rule.
    """

    boundary: int
    scores: np.ndarray
    n_valid_rows: int
    fit_ok: bool
    n_zeroed_rows: int = field(default=0)


class FeatureFrames:
    """Per-boundary causal feature frames with a cache.

    Args:
        series: (N, R) raw predictor series (regional extents or the
            scalar pan-Arctic extent), float64.
        months: (N,) calendar month of every record position.
        mode: ``raw`` / ``sigma_only`` / ``m3`` (see module docstring).
        pca: ``identity`` or ``pca5``.
        extra: optional (N, X) extra raw series appended to the frame
            scores after the decomposition (e.g. teleconnection indexes).
    """

    def __init__(self,
                 series: np.ndarray,
                 months: np.ndarray,
                 mode: str = "raw",
                 pca: str = "identity",
                 extra: Optional[np.ndarray] = None) -> None:
        if mode not in ("raw", "sigma_only", "m3"):
            raise ValueError(f"unknown predictor mode: {mode!r}")
        if pca not in ("identity", "pca5"):
            raise ValueError(f"unknown decomposition: {pca!r}")
        self.series = np.asarray(series, dtype=np.float64)
        self.months = np.asarray(months)
        self.mode = mode
        self.pca = pca
        self.extra = extra
        self._occurrences = {
            m: np.where(self.months == m)[0] for m in range(1, 13)}
        self._cache: dict[int, Frame] = {}

    def frame(self, boundary: int) -> Frame:
        """Frame of the case whose issue month IS ``boundary``: rows
        ``t <= boundary``, trailing climatology strictly before it, PCA
        fitted on the valid prefix rows."""
        cached = self._cache.get(boundary)
        if cached is not None:
            return cached

        n_series = self.series.shape[1]
        counts = np.zeros(13, dtype=int)
        means = np.zeros((13, n_series))
        stds = np.zeros((13, n_series))
        for m in range(1, 13):
            occ = self._occurrences[m]
            k = int(np.searchsorted(occ, boundary))  # strictly before b
            counts[m] = k
            n30 = min(M3_WINDOW, k)
            if n30 == 0:
                continue
            segment = self.series[occ[k - n30:k]]
            means[m] = segment.mean(axis=0)
            stds[m] = np.sqrt(segment.var(axis=0))  # ddof = 0

        rows = slice(0, boundary + 1)
        valid_per_month = counts >= MIN_POINTS
        valid_rows = valid_per_month[self.months[rows]]

        z = np.zeros((boundary + 1, n_series), dtype=np.float64)
        if self.mode == "m3":
            numerator = self.series[rows] - means[self.months[rows]]
        elif self.mode == "sigma_only":
            numerator = self.series[rows].copy()
        else:  # raw
            z = self.series[rows].copy()
        if self.mode in ("m3", "sigma_only"):
            sigma = stds[self.months[rows]]
            # sigma == 0 (constant trailing window) makes the
            # standardization undefined: the feature is zeroed by
            # convention (invalid rows are zeroed below anyway).
            safe = np.where(sigma > 0, sigma, 1.0)
            z = np.where(sigma > 0, numerator / safe, 0.0)
            z[~valid_rows] = 0.0

        n_valid = int(valid_rows.sum())
        fit_ok = True if self.mode == "raw" else n_valid >= MIN_FIT_ROWS

        if self.pca == "pca5":
            if fit_ok:
                model = fit_pca(z[valid_rows])
                scores = model.project(z)
            else:
                scores = np.zeros((boundary + 1, PCA_DIM))
        else:  # identity
            scores = z[:, :1]
        if self.extra is not None:
            scores = np.concatenate([scores, self.extra[rows]], axis=1)

        frame = Frame(boundary=boundary, scores=scores, n_valid_rows=n_valid,
                      fit_ok=fit_ok, n_zeroed_rows=int((~valid_rows).sum()))
        self._cache[boundary] = frame
        return frame
