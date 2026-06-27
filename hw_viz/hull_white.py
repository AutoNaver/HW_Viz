from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hw_viz.curve import CurveData


@dataclass(frozen=True)
class SimulationConfig:
    mean_reversion_a: float = 0.12
    volatility_sigma: float = 0.01
    simulation_horizon_years: float = 10.0
    time_step_months: int = 1
    num_paths: int = 1000
    random_seed: int = 7

    @property
    def dt(self) -> float:
        return self.time_step_months / 12.0

    def validate(self) -> None:
        if self.mean_reversion_a <= 0.0:
            raise ValueError("mean_reversion_a must be positive.")
        if self.volatility_sigma < 0.0:
            raise ValueError("volatility_sigma must be non-negative.")
        if self.simulation_horizon_years <= 0.0:
            raise ValueError("simulation_horizon_years must be positive.")
        if self.time_step_months <= 0:
            raise ValueError("time_step_months must be positive.")
        if self.num_paths <= 0:
            raise ValueError("num_paths must be positive.")


@dataclass(frozen=True)
class SimulationResult:
    time_grid: np.ndarray
    maturity_grid: np.ndarray
    short_rate_paths: np.ndarray
    yield_curves: np.ndarray
    alpha_curve: np.ndarray
    theta_curve: np.ndarray
    x_paths: np.ndarray

    def terminal_distribution(self, horizon_time: float) -> np.ndarray:
        index = int(np.argmin(np.abs(self.time_grid - horizon_time)))
        return self.short_rate_paths[:, index]


def standard_maturity_grid(curve: CurveData, horizon: float) -> np.ndarray:
    base = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    maturities = np.unique(np.concatenate([curve.maturities, base]))
    return maturities[maturities <= max(10.0, horizon)]


def alpha_from_curve(curve: CurveData, times: np.ndarray, a: float, sigma: float) -> np.ndarray:
    """Central tendency E[r(t)] of the simulated short rate.

    Textbook Hull-White pins this to the instantaneous forward ``f(0, t)``. We
    instead track the drawn zero curve ``z(t)`` (plus the usual variance term),
    so the simulated short-rate term structure follows the curve the user draws.
    This matters when an edit makes the curve steep: ``f(0, t) = z(t) + t z'(t)``
    overshoots and the short-rate distribution would otherwise snap back away
    from the drawn level. The trade-off is a minor loss of exact arbitrage-free
    repricing for ``t > 0`` (the affine yield reconstruction still uses the
    market discount curve), which is acceptable for this interactive tool.
    """
    base = curve.zero_rate(times)
    correction = np.zeros_like(times, dtype=float)
    if sigma > 0.0:
        correction = (sigma**2 / (2.0 * a**2)) * (1.0 - np.exp(-a * times)) ** 2
    return base + correction


def theta_from_curve(curve: CurveData, times: np.ndarray, a: float, sigma: float) -> np.ndarray:
    safe = np.maximum(times, curve.maturities[0])
    second = curve.interpolator.derivative(2)(safe)
    f = curve.instantaneous_forward(safe)
    f_prime = 2.0 * np.asarray(curve.derivative(safe), dtype=float) + safe * np.asarray(second, dtype=float)
    theta = f_prime + a * f
    if sigma > 0.0:
        theta = theta + (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * safe))
    return theta


def b_factor(a: float, t: np.ndarray | float, T: np.ndarray | float) -> np.ndarray:
    return (1.0 - np.exp(-a * (np.asarray(T) - np.asarray(t)))) / a


def bond_price(curve: CurveData, a: float, sigma: float, t: float, T: np.ndarray, short_rate: np.ndarray) -> np.ndarray:
    t_array = np.asarray(t, dtype=float)
    T_array = np.asarray(T, dtype=float)
    B = b_factor(a, t_array, T_array)
    p0_T = curve.discount_factor(T_array)
    p0_t = curve.discount_factor(t_array)
    f0_t = curve.instantaneous_forward(np.array([max(t, curve.maturities[0])], dtype=float))[0]
    variance_adjustment = (sigma**2 / (4.0 * a)) * (1.0 - np.exp(-2.0 * a * t_array)) * (B**2)
    log_a = np.log(p0_T / p0_t) + B * f0_t - variance_adjustment
    return np.exp(log_a - B * np.asarray(short_rate, dtype=float))


def yield_curves_from_short_rates(
    curve: CurveData,
    config: SimulationConfig,
    time_grid: np.ndarray,
    short_rate_paths: np.ndarray,
    maturity_grid: np.ndarray,
) -> np.ndarray:
    yields = np.zeros((short_rate_paths.shape[0], time_grid.size, maturity_grid.size), dtype=float)
    for time_index, t in enumerate(time_grid):
        T = t + maturity_grid
        prices = bond_price(
            curve=curve,
            a=config.mean_reversion_a,
            sigma=config.volatility_sigma,
            t=float(t),
            T=T,
            short_rate=short_rate_paths[:, time_index][:, None],
        )
        yields[:, time_index, :] = -np.log(np.maximum(prices, 1e-12)) / maturity_grid
    return yields


def simulate_hull_white_paths(curve: CurveData, config: SimulationConfig) -> SimulationResult:
    config.validate()
    dt = config.dt
    step_count = int(round(config.simulation_horizon_years / dt))
    time_grid = np.linspace(0.0, step_count * dt, step_count + 1)
    maturity_grid = standard_maturity_grid(curve, config.simulation_horizon_years)
    alpha_curve = alpha_from_curve(curve, time_grid, config.mean_reversion_a, config.volatility_sigma)
    theta_curve = theta_from_curve(curve, time_grid, config.mean_reversion_a, config.volatility_sigma)

    rng = np.random.default_rng(config.random_seed)
    x_paths = np.zeros((config.num_paths, time_grid.size), dtype=float)
    phi = np.exp(-config.mean_reversion_a * dt)
    q = (config.volatility_sigma**2 / (2.0 * config.mean_reversion_a)) * (1.0 - np.exp(-2.0 * config.mean_reversion_a * dt))
    shocks = rng.normal(size=(config.num_paths, step_count))
    for index in range(step_count):
        x_paths[:, index + 1] = phi * x_paths[:, index] + np.sqrt(max(q, 0.0)) * shocks[:, index]

    short_rate_paths = x_paths + alpha_curve[None, :]
    yield_curves = yield_curves_from_short_rates(curve, config, time_grid, short_rate_paths, maturity_grid)

    return SimulationResult(
        time_grid=time_grid,
        maturity_grid=maturity_grid,
        short_rate_paths=short_rate_paths,
        yield_curves=yield_curves,
        alpha_curve=alpha_curve,
        theta_curve=theta_curve,
        x_paths=x_paths,
    )
