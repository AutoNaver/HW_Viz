from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


TENOR_PATTERN = re.compile(r"^\s*(\d+)\s*([DdWwMmYy])\s*$")


def parse_tenor(tenor: str) -> float:
    match = TENOR_PATTERN.match(str(tenor))
    if not match:
        raise ValueError(f"Invalid tenor value: {tenor!r}")

    value = float(match.group(1))
    unit = match.group(2).upper()
    factors = {
        "D": 1.0 / 365.0,
        "W": 7.0 / 365.0,
        "M": 1.0 / 12.0,
        "Y": 1.0,
    }
    return value * factors[unit]


def _normalize_rate_series(rates: pd.Series) -> np.ndarray:
    values = pd.to_numeric(rates, errors="raise").astype(float).to_numpy()
    if np.nanmax(np.abs(values)) > 1.0:
        values = values / 100.0
    return np.log1p(values)


@dataclass(frozen=True)
class CurveData:
    input_frame: pd.DataFrame
    maturities: np.ndarray
    zero_rates: np.ndarray
    discount_factors: np.ndarray
    interpolator: PchipInterpolator
    derivative: PchipInterpolator

    def zero_rate(self, maturity: float | np.ndarray) -> np.ndarray:
        maturity_array = np.asarray(maturity, dtype=float)
        safe = np.maximum(maturity_array, self.maturities[0])
        return np.asarray(self.interpolator(safe), dtype=float)

    def discount_factor(self, maturity: float | np.ndarray) -> np.ndarray:
        maturity_array = np.asarray(maturity, dtype=float)
        result = np.ones_like(maturity_array, dtype=float)
        positive = maturity_array > 0.0
        if np.any(positive):
            safe = maturity_array[positive]
            result[positive] = np.exp(-self.zero_rate(safe) * safe)
        return result

    def instantaneous_forward(self, maturity: float | np.ndarray) -> np.ndarray:
        maturity_array = np.asarray(maturity, dtype=float)
        safe = np.maximum(maturity_array, self.maturities[0])
        z = np.asarray(self.interpolator(safe), dtype=float)
        dz = np.asarray(self.derivative(safe), dtype=float)
        forwards = z + safe * dz
        near_zero = maturity_array <= 0.0
        if np.any(near_zero):
            forwards = np.asarray(forwards, dtype=float)
            forwards[near_zero] = self.instantaneous_forward(self.maturities[0])[()]
        return forwards


def default_curve_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenor": ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "20Y", "30Y"],
            "rate": [4.10, 4.05, 4.00, 3.90, 3.70, 3.35, 3.15, 3.05, 3.00],
        }
    )


def apply_curve_drag(
    maturities: np.ndarray,
    rates: np.ndarray,
    index: int,
    new_rate: float,
    smoothing: bool = True,
    length_scale: float = 0.8,
) -> np.ndarray:
    """Return updated rates after dragging a single tenor point.

    The dragged point at ``index`` is moved to ``new_rate``. When ``smoothing``
    is enabled, the change propagates to neighbouring tenors with a Gaussian
    weight in log-maturity space: nearby short tenors follow the drag partially
    while the long end (far away in log-maturity) stays essentially fixed, which
    keeps the curve smooth after an edit. With ``smoothing`` disabled only the
    grabbed point moves.

    ``length_scale`` is the Gaussian width in natural-log-of-maturity units; a
    larger value lets more distant tenors follow the drag.
    """

    maturities = np.asarray(maturities, dtype=float)
    rates = np.asarray(rates, dtype=float).copy()
    if not (0 <= index < rates.size):
        raise IndexError(f"index {index} is out of range for {rates.size} rates")

    if not smoothing:
        rates[index] = float(new_rate)
        return rates

    delta = float(new_rate) - rates[index]
    log_maturities = np.log(np.maximum(maturities, 1e-12))
    distance = log_maturities - log_maturities[index]
    weights = np.exp(-(distance**2) / (2.0 * length_scale**2))
    return rates + weights * delta


def build_curve(frame: pd.DataFrame) -> CurveData:
    if not {"tenor", "rate"}.issubset(frame.columns):
        raise ValueError("Curve input must contain tenor and rate columns.")

    working = frame.loc[:, ["tenor", "rate"]].copy()
    if working.empty:
        raise ValueError("Curve input is empty.")

    working["tenor"] = working["tenor"].astype(str).str.strip()
    working["maturity"] = working["tenor"].map(parse_tenor)
    working["rate"] = pd.to_numeric(working["rate"], errors="raise")
    working = working.sort_values("maturity", kind="stable").reset_index(drop=True)

    if working["maturity"].duplicated().any():
        raise ValueError("Curve tenors must map to unique maturities.")
    if (working["maturity"] <= 0.0).any():
        raise ValueError("Curve maturities must be positive.")
    if not np.all(np.diff(working["maturity"]) > 0.0):
        raise ValueError("Curve maturities must be strictly increasing.")

    maturities = working["maturity"].to_numpy(dtype=float)
    zero_rates = _normalize_rate_series(working["rate"])
    discount_factors = np.exp(-zero_rates * maturities)
    interpolator = PchipInterpolator(maturities, zero_rates, extrapolate=True)
    derivative = interpolator.derivative()

    return CurveData(
        input_frame=working.loc[:, ["tenor", "rate"]].copy(),
        maturities=maturities,
        zero_rates=zero_rates,
        discount_factors=discount_factors,
        interpolator=interpolator,
        derivative=derivative,
    )
