from __future__ import annotations

import numpy as np

from hw_viz.curve import CurveData
from hw_viz.distribution import (
    DISTRIBUTION_ANCHOR_LEVELS,
    DistributionEdit,
    FittedDistribution,
    fit_distribution_from_quantiles,
)
from hw_viz.hull_white import SimulationConfig, SimulationResult, yield_curves_from_short_rates


def regenerate_scenarios_surface(
    curve: CurveData,
    config: SimulationConfig,
    base_result: SimulationResult,
    anchor_surface: np.ndarray,
    edited_mask: np.ndarray,
    anchor_levels: np.ndarray = DISTRIBUTION_ANCHOR_LEVELS,
) -> SimulationResult:
    """Regenerate scenarios honouring an edited distribution surface over horizons.

    For every time index flagged in ``edited_mask`` the simulated short-rate
    cross-section is remapped to the target distribution fitted from that
    horizon's quantile anchors, using a rank (copula-preserving) transform: each
    path keeps its relative position while the marginal is forced to the target.
    Time indices that were not edited keep the base simulation exactly. Because
    edits propagate smoothly across adjacent horizons, the remapped region tapers
    in and out and paths stay smooth.

    This honours the whole edited surface rather than a single horizon, which is
    what the horizon-aware editor produces.
    """

    anchor_surface = np.asarray(anchor_surface, dtype=float)
    edited_mask = np.asarray(edited_mask, dtype=bool)
    time_grid = base_result.time_grid
    short_rate = base_result.short_rate_paths.copy()
    num_paths = short_rate.shape[0]
    uniform_ranks_base = (np.arange(num_paths) + 0.5) / num_paths

    for index in range(1, time_grid.size):
        if not edited_mask[index]:
            continue
        target = fit_distribution_from_quantiles(anchor_levels, anchor_surface[index])
        column = short_rate[:, index]
        order = np.argsort(column, kind="stable")
        ranks = np.empty(num_paths, dtype=float)
        ranks[order] = uniform_ranks_base
        ranks = np.clip(ranks, 1e-4, 1.0 - 1e-4)
        short_rate[:, index] = np.asarray(target.quantile_interpolator(ranks), dtype=float)

    x_paths = short_rate - base_result.alpha_curve[None, :]
    yield_curves = yield_curves_from_short_rates(
        curve, config, time_grid, short_rate, base_result.maturity_grid
    )

    return SimulationResult(
        time_grid=time_grid,
        maturity_grid=base_result.maturity_grid,
        short_rate_paths=short_rate,
        yield_curves=yield_curves,
        alpha_curve=base_result.alpha_curve,
        theta_curve=base_result.theta_curve,
        x_paths=x_paths,
    )


def regenerate_scenarios(
    curve: CurveData,
    config: SimulationConfig,
    base_result: SimulationResult,
    edit: DistributionEdit,
    fitted: FittedDistribution,
) -> SimulationResult:
    dt = config.dt
    time_grid = base_result.time_grid
    maturity_grid = base_result.maturity_grid
    horizon_index = int(np.argmin(np.abs(time_grid - edit.horizon_time)))
    horizon_time = float(time_grid[horizon_index])
    rng = np.random.default_rng(config.random_seed + 1_000)

    target_rates = fitted.sample(config.num_paths, random_seed=config.random_seed + 2_000)
    target_x = target_rates - base_result.alpha_curve[horizon_index]

    step_count = time_grid.size - 1
    phi = np.exp(-config.mean_reversion_a * dt)
    q = (config.volatility_sigma**2 / (2.0 * config.mean_reversion_a)) * (1.0 - np.exp(-2.0 * config.mean_reversion_a * dt))
    q_sqrt = np.sqrt(max(q, 0.0))

    x_paths = np.zeros((config.num_paths, time_grid.size), dtype=float)
    shocks = rng.normal(size=(config.num_paths, step_count))
    for index in range(horizon_index):
        x_paths[:, index + 1] = phi * x_paths[:, index] + q_sqrt * shocks[:, index]

    current_terminal = x_paths[:, horizon_index]
    if horizon_index > 0:
        variances = (config.volatility_sigma**2 / (2.0 * config.mean_reversion_a)) * (
            1.0 - np.exp(-2.0 * config.mean_reversion_a * time_grid[: horizon_index + 1])
        )
        terminal_variance = max(variances[-1], 1e-12)
        beta = variances * np.exp(-config.mean_reversion_a * (horizon_time - time_grid[: horizon_index + 1])) / terminal_variance
        x_paths[:, : horizon_index + 1] += beta[None, :] * (target_x - current_terminal)[:, None]
    x_paths[:, horizon_index] = target_x

    for index in range(horizon_index, step_count):
        x_paths[:, index + 1] = phi * x_paths[:, index] + q_sqrt * shocks[:, index]

    short_rate_paths = x_paths + base_result.alpha_curve[None, :]
    yield_curves = yield_curves_from_short_rates(curve, config, time_grid, short_rate_paths, maturity_grid)

    return SimulationResult(
        time_grid=time_grid,
        maturity_grid=maturity_grid,
        short_rate_paths=short_rate_paths,
        yield_curves=yield_curves,
        alpha_curve=base_result.alpha_curve,
        theta_curve=base_result.theta_curve,
        x_paths=x_paths,
    )
