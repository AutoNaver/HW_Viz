from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hw_viz.curve import apply_curve_drag, build_curve, default_curve_frame, parse_tenor
from hw_viz.distribution import (
    DISTRIBUTION_ANCHOR_LEVELS,
    QUANTILE_LEVELS,
    build_anchor_surface,
    build_distribution_edit,
    build_distribution_edit_from_anchors,
    empirical_quantiles,
    fit_distribution,
    fit_distribution_from_quantiles,
    propagate_horizon_drag,
)
from hw_viz.exports import metadata_json, short_rate_frame, yield_curve_frame
from hw_viz.hull_white import SimulationConfig, simulate_hull_white_paths
from hw_viz.regeneration import regenerate_scenarios, regenerate_scenarios_surface


def sample_curve_frame() -> pd.DataFrame:
    return default_curve_frame()


def small_config(seed: int = 11, num_paths: int = 600) -> SimulationConfig:
    return SimulationConfig(
        mean_reversion_a=0.15,
        volatility_sigma=0.01,
        simulation_horizon_years=5.0,
        time_step_months=1,
        num_paths=num_paths,
        random_seed=seed,
    )


def test_parse_tenor_supports_standard_units() -> None:
    assert np.isclose(parse_tenor("6M"), 0.5)
    assert np.isclose(parse_tenor("2Y"), 2.0)
    assert np.isclose(parse_tenor("4W"), 28.0 / 365.0)


def test_build_curve_rejects_duplicate_maturities() -> None:
    frame = pd.DataFrame({"tenor": ["6M", "6M"], "rate": [4.0, 4.1]})
    try:
        build_curve(frame)
    except ValueError as exc:
        assert "unique maturities" in str(exc)
    else:
        raise AssertionError("Expected duplicate maturity validation to fail.")


def test_curve_discount_factors_decrease_monotonically() -> None:
    curve = build_curve(sample_curve_frame())
    discounts = curve.discount_factor(curve.maturities)
    assert np.all(np.diff(discounts) < 0.0)


def test_theta_is_finite_on_simulation_grid() -> None:
    curve = build_curve(sample_curve_frame())
    result = simulate_hull_white_paths(curve, small_config())
    assert np.all(np.isfinite(result.theta_curve))


def test_short_rate_drift_tracks_drawn_zero_curve() -> None:
    # A steep front (short end dragged down, long end fixed) must not make the
    # mean short rate snap back to the long-end level: E[r(t)] should follow the
    # drawn zero curve, not the overshooting instantaneous forward.
    frame = sample_curve_frame()
    tenors = frame["tenor"].tolist()
    maturities = np.array([parse_tenor(t) for t in tenors])
    rates = frame["rate"].to_numpy(float)
    dragged = apply_curve_drag(maturities, rates, index=0, new_rate=2.0, smoothing=True)
    curve = build_curve(pd.DataFrame({"tenor": tenors, "rate": dragged}))
    result = simulate_hull_white_paths(curve, small_config(num_paths=4000))

    mean_short = result.short_rate_paths.mean(axis=0)
    drawn_zero = curve.zero_rate(result.time_grid)
    # Mean short rate stays close to the drawn zero curve (variance term is tiny).
    assert np.allclose(mean_short, drawn_zero, atol=3e-3)
    # In particular the very front reflects the drag rather than snapping to ~4%.
    assert mean_short[0] < 0.025


def test_simulation_is_seed_reproducible() -> None:
    curve = build_curve(sample_curve_frame())
    result_a = simulate_hull_white_paths(curve, small_config(seed=17))
    result_b = simulate_hull_white_paths(curve, small_config(seed=17))
    assert np.allclose(result_a.short_rate_paths, result_b.short_rate_paths)
    assert np.allclose(result_a.yield_curves, result_b.yield_curves)


def test_initial_model_yields_match_input_curve() -> None:
    curve = build_curve(sample_curve_frame())
    result = simulate_hull_white_paths(curve, small_config())
    initial_yields = result.yield_curves[0, 0]
    target_yields = np.array([curve.zero_rate(maturity)[()] for maturity in result.maturity_grid])
    assert np.allclose(initial_yields, target_yields, atol=2e-3)


def test_curve_drag_smoothing_propagates_to_neighbours_and_fixes_long_end() -> None:
    maturities = np.array([1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    rates = np.array([4.10, 4.05, 4.00, 3.90, 3.70, 3.35, 3.15, 3.00])

    bumped = apply_curve_drag(maturities, rates, index=0, new_rate=4.60, smoothing=True)

    # Dragged point lands exactly on the requested value.
    assert np.isclose(bumped[0], 4.60)
    # Nearby short tenor follows partially (between original and the full bump).
    assert rates[1] < bumped[1] < bumped[0]
    # Closer tenors follow more strongly than farther ones.
    assert (bumped[1] - rates[1]) > (bumped[2] - rates[2]) > 0.0
    # The long end is essentially unchanged.
    assert np.isclose(bumped[-1], rates[-1], atol=1e-3)


def test_curve_drag_without_smoothing_moves_only_target() -> None:
    maturities = np.array([1.0 / 12.0, 0.25, 0.5, 1.0])
    rates = np.array([4.10, 4.05, 4.00, 3.90])
    moved = apply_curve_drag(maturities, rates, index=1, new_rate=3.50, smoothing=False)
    assert np.isclose(moved[1], 3.50)
    untouched = np.delete(moved, 1)
    assert np.allclose(untouched, np.delete(rates, 1))


def test_fit_distribution_from_quantiles_is_well_formed() -> None:
    levels = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    values = np.array([0.005, 0.012, 0.022, 0.030, 0.038, 0.050, 0.060])
    fitted = fit_distribution_from_quantiles(levels, values)
    assert np.all(np.diff(fitted.value_grid) > 0.0)
    assert np.all(np.diff(fitted.cdf_grid) >= 0.0)
    assert np.all(fitted.pdf_grid >= 0.0)
    assert np.isclose(np.trapezoid(fitted.pdf_grid, fitted.value_grid), 1.0, atol=2e-3)
    # The fitted distribution reproduces the dragged anchors at their quantiles.
    sampled = fitted.sample(40000, random_seed=3)
    recovered = empirical_quantiles(sampled, levels)
    assert np.allclose(recovered, values, atol=4e-3)


def test_build_distribution_edit_from_anchors_extracts_standard_quantiles() -> None:
    levels = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    values = np.array([0.005, 0.012, 0.022, 0.030, 0.038, 0.050, 0.060])
    edit = build_distribution_edit_from_anchors(levels, values, horizon_time=3.0)
    assert np.allclose(edit.quantile_levels, QUANTILE_LEVELS)
    expected = np.interp(QUANTILE_LEVELS, levels, values)
    assert np.allclose(edit.quantile_values, expected)
    assert edit.left_tail_scale == 1.0 and edit.right_tail_scale == 1.0


def test_propagate_horizon_drag_spreads_to_adjacent_horizons() -> None:
    times = np.linspace(0.0, 5.0, 11)
    levels = DISTRIBUTION_ANCHOR_LEVELS
    surface = np.tile(np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]), (times.size, 1))
    target_index = 5
    level_index = 3  # the 50% anchor

    updated = propagate_horizon_drag(
        times, surface, target_index, level_index, new_value=0.055, smoothing=True, length_scale=1.0
    )

    original = surface[target_index, level_index]
    # Dragged horizon moves the full amount.
    assert np.isclose(updated[target_index, level_index], 0.055)
    # Adjacent horizons follow partially in the same direction.
    moved_neighbour = updated[target_index + 1, level_index] - original
    moved_far = updated[0, level_index] - original
    assert moved_neighbour > 0.0
    assert moved_neighbour > moved_far
    # Far horizon barely moves.
    assert abs(moved_far) < abs(updated[target_index, level_index] - original)
    # Every row stays a valid (strictly increasing) quantile vector.
    assert np.all(np.diff(updated, axis=1) > 0.0)


def test_propagate_horizon_drag_without_spreading_is_local() -> None:
    times = np.linspace(0.0, 5.0, 11)
    surface = np.tile(np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]), (times.size, 1))
    updated = propagate_horizon_drag(times, surface, 5, 2, new_value=0.035, smoothing=False)
    assert np.isclose(updated[5, 2], 0.035)
    others = np.delete(updated, 5, axis=0)
    assert np.allclose(others, np.delete(surface, 5, axis=0))


def test_surface_regeneration_matches_targets_and_preserves_unedited() -> None:
    curve = build_curve(sample_curve_frame())
    config = small_config(seed=29, num_paths=4000)
    base = simulate_hull_white_paths(curve, config)
    surface = build_anchor_surface(base.short_rate_paths, DISTRIBUTION_ANCHOR_LEVELS)

    edited_index = int(np.argmin(np.abs(base.time_grid - 3.0)))
    surface[edited_index] = surface[edited_index] + 0.004  # shift the whole horizon up
    edited_mask = np.zeros(base.time_grid.size, dtype=bool)
    edited_mask[edited_index] = True

    regenerated = regenerate_scenarios_surface(curve, config, base, surface, edited_mask, DISTRIBUTION_ANCHOR_LEVELS)

    # The edited horizon's marginal tracks the requested anchors.
    recovered = empirical_quantiles(regenerated.short_rate_paths[:, edited_index], DISTRIBUTION_ANCHOR_LEVELS)
    assert np.allclose(recovered, surface[edited_index], atol=3e-3)
    # Unedited horizons are untouched.
    untouched = np.delete(np.arange(base.time_grid.size), edited_index)
    assert np.allclose(
        regenerated.short_rate_paths[:, untouched], base.short_rate_paths[:, untouched]
    )


def test_distribution_fit_is_well_formed() -> None:
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=0.03, scale=0.01, size=3000)
    edit = build_distribution_edit(samples=samples, horizon_time=3.0, left_tail_scale=1.5, right_tail_scale=1.25)
    fitted = fit_distribution(edit)
    assert np.all(np.diff(fitted.value_grid) > 0.0)
    assert np.all(np.diff(fitted.cdf_grid) >= 0.0)
    assert np.all(fitted.pdf_grid >= 0.0)
    assert np.isclose(np.trapezoid(fitted.pdf_grid, fitted.value_grid), 1.0, atol=2e-3)


def test_regenerated_terminal_quantiles_track_edited_target() -> None:
    curve = build_curve(sample_curve_frame())
    config = small_config(seed=23, num_paths=4000)
    base = simulate_hull_white_paths(curve, config)
    horizon_time = 3.0
    base_samples = base.terminal_distribution(horizon_time)
    base_quantiles = empirical_quantiles(base_samples, QUANTILE_LEVELS)
    edited_quantiles = base_quantiles + np.array([-0.005, -0.002, 0.001, 0.003, 0.006])
    edit = build_distribution_edit(
        samples=base_samples,
        horizon_time=horizon_time,
        quantile_values=edited_quantiles,
        left_tail_scale=1.25,
        right_tail_scale=1.5,
    )
    fitted = fit_distribution(edit)
    regenerated = regenerate_scenarios(curve, config, base, edit, fitted)
    regenerated_quantiles = empirical_quantiles(regenerated.terminal_distribution(horizon_time), QUANTILE_LEVELS)
    assert np.allclose(regenerated_quantiles, edit.quantile_values, atol=2.5e-3)


def test_end_to_end_export_workflow_produces_artifacts() -> None:
    curve_frame = sample_curve_frame()
    curve = build_curve(curve_frame)
    config = small_config(seed=31, num_paths=300)
    base = simulate_hull_white_paths(curve, config)
    horizon_time = 2.0
    samples = base.terminal_distribution(horizon_time)
    edit = build_distribution_edit(samples=samples, horizon_time=horizon_time)
    fitted = fit_distribution(edit)
    regenerated = regenerate_scenarios(curve, config, base, edit, fitted)

    short_rates = short_rate_frame(regenerated, "regenerated")
    yields = yield_curve_frame(regenerated, "regenerated")
    metadata = json.loads(metadata_json(curve_frame, config, edit, "regenerated"))

    assert not short_rates.empty
    assert not yields.empty
    assert metadata["config"]["num_paths"] == config.num_paths
    assert len(metadata["curve"]) == len(curve_frame)
