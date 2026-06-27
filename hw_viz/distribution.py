from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator


QUANTILE_LEVELS = np.array([0.05, 0.25, 0.50, 0.75, 0.95], dtype=float)

# Quantile anchors exposed in the draggable distribution editor. The outer
# 0.01 / 0.99 anchors give the user direct control of the tails.
DISTRIBUTION_ANCHOR_LEVELS = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], dtype=float)


def empirical_quantiles(samples: np.ndarray, levels: np.ndarray = QUANTILE_LEVELS) -> np.ndarray:
    return np.quantile(np.asarray(samples, dtype=float), np.asarray(levels, dtype=float))


def enforce_strict_increase(values: np.ndarray, minimum_step: float = 1e-6) -> np.ndarray:
    adjusted = np.asarray(values, dtype=float).copy()
    for index in range(1, adjusted.size):
        adjusted[index] = max(adjusted[index], adjusted[index - 1] + minimum_step)
    return adjusted


@dataclass(frozen=True)
class DistributionEdit:
    horizon_time: float
    quantile_levels: np.ndarray
    quantile_values: np.ndarray
    left_tail_scale: float
    right_tail_scale: float


@dataclass(frozen=True)
class FittedDistribution:
    probability_grid: np.ndarray
    value_grid: np.ndarray
    cdf_grid: np.ndarray
    pdf_grid: np.ndarray
    quantile_interpolator: PchipInterpolator
    cdf_interpolator: PchipInterpolator

    def sample(self, size: int, random_seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(random_seed)
        uniforms = rng.uniform(1e-4, 1.0 - 1e-4, size=size)
        return np.asarray(self.quantile_interpolator(uniforms), dtype=float)

    def pdf(self, values: np.ndarray) -> np.ndarray:
        return np.interp(values, self.value_grid, self.pdf_grid, left=0.0, right=0.0)

    def cdf(self, values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, self.value_grid[0], self.value_grid[-1])
        return np.asarray(self.cdf_interpolator(clipped), dtype=float)


def build_distribution_edit(
    samples: np.ndarray,
    horizon_time: float,
    quantile_levels: np.ndarray = QUANTILE_LEVELS,
    quantile_values: np.ndarray | None = None,
    left_tail_scale: float = 1.0,
    right_tail_scale: float = 1.0,
) -> DistributionEdit:
    levels = np.asarray(quantile_levels, dtype=float)
    values = empirical_quantiles(samples, levels) if quantile_values is None else np.asarray(quantile_values, dtype=float)
    values = enforce_strict_increase(values)
    return DistributionEdit(
        horizon_time=float(horizon_time),
        quantile_levels=levels,
        quantile_values=values,
        left_tail_scale=float(left_tail_scale),
        right_tail_scale=float(right_tail_scale),
    )


def fit_distribution(edit: DistributionEdit, grid_size: int = 801) -> FittedDistribution:
    levels = np.asarray(edit.quantile_levels, dtype=float)
    values = enforce_strict_increase(edit.quantile_values)

    left_span = max(values[1] - values[0], 1e-4)
    right_span = max(values[-1] - values[-2], 1e-4)
    left_value = values[0] - edit.left_tail_scale * left_span
    right_value = values[-1] + edit.right_tail_scale * right_span

    full_levels = np.concatenate(([1e-4, 0.01], levels, [0.99, 1.0 - 1e-4]))
    full_values = np.concatenate(
        (
            [left_value - 0.5 * left_span, left_value],
            values,
            [right_value, right_value + 0.5 * right_span],
        )
    )
    return _finalize_quantile_fit(full_levels, full_values, grid_size)


def fit_distribution_from_quantiles(
    levels: np.ndarray,
    values: np.ndarray,
    grid_size: int = 801,
) -> FittedDistribution:
    """Fit a smooth distribution directly from dragged quantile anchors.

    ``levels`` are probability levels in ``(0, 1)`` (already including the tail
    anchors the user controls, e.g. ``0.01`` and ``0.99``) and ``values`` are
    the corresponding short-rate values. Tiny epsilon endpoints are appended so
    the inverse-CDF spline covers the full unit interval. The downstream
    density/CDF construction is shared with :func:`fit_distribution`.
    """

    levels = np.asarray(levels, dtype=float)
    values = enforce_strict_increase(np.asarray(values, dtype=float))
    if levels.size != values.size or levels.size < 3:
        raise ValueError("levels and values must have matching length >= 3")

    left_span = max(values[1] - values[0], 1e-4)
    right_span = max(values[-1] - values[-2], 1e-4)
    full_levels = np.concatenate(([1e-4], levels, [1.0 - 1e-4]))
    full_values = np.concatenate(([values[0] - 0.5 * left_span], values, [values[-1] + 0.5 * right_span]))
    return _finalize_quantile_fit(full_levels, full_values, grid_size)


def build_distribution_edit_from_anchors(
    levels: np.ndarray,
    values: np.ndarray,
    horizon_time: float,
) -> DistributionEdit:
    """Build a :class:`DistributionEdit` from dragged quantile anchors.

    The standard reporting quantiles (:data:`QUANTILE_LEVELS`) are extracted from
    the dragged anchors so exports and metadata stay meaningful. Tail behaviour
    is now expressed directly through the anchor values, so the tail scales are
    left at their neutral value of ``1.0``.
    """

    levels = np.asarray(levels, dtype=float)
    values = enforce_strict_increase(np.asarray(values, dtype=float))
    quantile_values = np.interp(QUANTILE_LEVELS, levels, values)
    return DistributionEdit(
        horizon_time=float(horizon_time),
        quantile_levels=QUANTILE_LEVELS,
        quantile_values=quantile_values,
        left_tail_scale=1.0,
        right_tail_scale=1.0,
    )


def build_anchor_surface(
    short_rate_paths: np.ndarray,
    levels: np.ndarray = DISTRIBUTION_ANCHOR_LEVELS,
) -> np.ndarray:
    """Build the (num_times, num_levels) empirical quantile-anchor surface.

    Each row holds the empirical quantile values of the short-rate cross-section
    at the corresponding time index. This is the starting point for the
    horizon-aware distribution editor.
    """

    paths = np.asarray(short_rate_paths, dtype=float)
    levels = np.asarray(levels, dtype=float)
    surface = np.quantile(paths, levels, axis=0).T
    return np.ascontiguousarray(surface, dtype=float)


def propagate_horizon_drag(
    times: np.ndarray,
    surface: np.ndarray,
    time_index: int,
    level_index: int,
    new_value: float,
    smoothing: bool = True,
    length_scale: float = 1.0,
) -> np.ndarray:
    """Return an updated anchor surface after dragging one quantile point.

    The anchor at ``(time_index, level_index)`` is moved to ``new_value``. When
    ``smoothing`` is enabled the change propagates to adjacent horizons with a
    Gaussian weight in time (``length_scale`` in years), so neighbouring horizons
    follow the edit and the surface stays smooth across time. With ``smoothing``
    disabled only the grabbed horizon changes. Each affected row is kept strictly
    increasing across quantiles.
    """

    times = np.asarray(times, dtype=float)
    surface = np.asarray(surface, dtype=float).copy()
    delta = float(new_value) - surface[time_index, level_index]

    if smoothing:
        distance = times - times[time_index]
        weights = np.exp(-(distance**2) / (2.0 * max(length_scale, 1e-6) ** 2))
    else:
        weights = np.zeros(times.size, dtype=float)
        weights[time_index] = 1.0

    surface[:, level_index] = surface[:, level_index] + weights * delta
    for row in range(surface.shape[0]):
        if weights[row] != 0.0:
            surface[row] = enforce_strict_increase(surface[row])
    return surface


def _finalize_quantile_fit(full_levels: np.ndarray, full_values: np.ndarray, grid_size: int) -> FittedDistribution:
    full_values = enforce_strict_increase(full_values)
    quantile_interpolator = PchipInterpolator(full_levels, full_values, extrapolate=False)
    probability_grid = np.linspace(full_levels[0], full_levels[-1], grid_size)
    value_grid = np.asarray(quantile_interpolator(probability_grid), dtype=float)
    derivative = np.asarray(quantile_interpolator.derivative()(probability_grid), dtype=float)
    derivative = np.maximum(derivative, 1e-8)
    pdf_grid = 1.0 / derivative
    normalization = np.trapezoid(pdf_grid, value_grid)
    pdf_grid = pdf_grid / normalization
    cdf_grid = np.cumsum((pdf_grid[1:] + pdf_grid[:-1]) * np.diff(value_grid) / 2.0)
    cdf_grid = np.concatenate(([0.0], cdf_grid))
    cdf_grid = cdf_grid / cdf_grid[-1]
    cdf_interpolator = PchipInterpolator(value_grid, cdf_grid, extrapolate=True)

    return FittedDistribution(
        probability_grid=probability_grid,
        value_grid=value_grid,
        cdf_grid=cdf_grid,
        pdf_grid=pdf_grid,
        quantile_interpolator=quantile_interpolator,
        cdf_interpolator=cdf_interpolator,
    )
