# Architecture

## Overview

The project is organized around a thin UI layer and a small quantitative core:

- `app.py` orchestrates user input, charts, and exports.
- `hw_viz/` contains reusable model and data-transformation logic.
- `tests/` validates numerical consistency and workflow expectations.

The intended maintenance pattern is:

- UI concerns stay in `app.py`
- model math stays in `hw_viz/`
- behavior changes are captured by tests before refactors

## Data Flow

1. The user uploads, edits, or drags a `tenor` / `rate` curve. Drags are turned into a smoothed rate update by `hw_viz.curve.apply_curve_drag()`.
2. `hw_viz.curve.build_curve()` parses tenors, normalizes rates, and builds a monotone zero-curve interpolator.
3. `hw_viz.hull_white.simulate_hull_white_paths()` derives `alpha(t)` and `theta(t)`, simulates the centered OU state, and converts paths to model-implied yield curves. The central tendency `alpha(t)` tracks the drawn zero curve `z(t)` (not the instantaneous forward), so the simulated short-rate term structure follows the curve the user draws.
4. The UI maintains a distribution *surface* (`build_anchor_surface()`): a row of quantile anchors per time index. Dragging a quantile point at the selected horizon is applied via `propagate_horizon_drag()`, which can spread the edit smoothly to adjacent horizons (Gaussian in time).
5. `hw_viz.distribution.fit_distribution_from_quantiles()` turns each horizon's dragged quantile anchors into a monotone spline quantile function and a valid density/CDF representation; `build_distribution_edit_from_anchors()` records the standard reporting quantiles for the selected horizon.
6. `hw_viz.regeneration.regenerate_scenarios_surface()` remaps each edited horizon's simulated marginal to its target via a rank (copula-preserving) transform, leaving unedited horizons untouched. The single-horizon OU-bridge `regenerate_scenarios()` is retained for reference.
7. `hw_viz.exports` converts the regenerated result into flat export tables and metadata JSON.

## Module Responsibilities

## `hw_viz.curve`

- `parse_tenor()`: converts values such as `1M`, `6M`, `5Y` into year fractions
- `build_curve()`: validates the input frame and builds `CurveData`
- `apply_curve_drag()`: applies a single dragged tenor change, optionally propagating it smoothly to neighbouring tenors via a Gaussian kernel in log-maturity
- `CurveData`: exposes zero rates, discount factors, and instantaneous forwards

This module should remain the only place that knows how raw curve input is interpreted.

## `hw_viz.hull_white`

- `SimulationConfig`: simulation parameters and validation
- `SimulationResult`: container for time grid, path outputs, and derived arrays
- `simulate_hull_white_paths()`: main simulation entrypoint
- `bond_price()` and `yield_curves_from_short_rates()`: pricing layer used for curve generation

If the model is extended later, add new functions here first and keep the UI unaware of implementation details.

## `hw_viz.distribution`

- `DistributionEdit`: user intent at the chosen horizon
- `FittedDistribution`: fitted quantile/CDF/PDF representation
- `fit_distribution()`: converts quantiles plus tail scales into a smooth valid distribution
- `fit_distribution_from_quantiles()`: converts dragged quantile anchors (including explicit tail levels) directly into a smooth valid distribution
- `build_distribution_edit_from_anchors()`: records the standard reporting quantiles from dragged anchors
- `build_anchor_surface()`: builds the per-horizon empirical quantile-anchor surface
- `propagate_horizon_drag()`: applies a single quantile-point drag, optionally spreading it to adjacent horizons with a Gaussian-in-time weight

This module defines the contract between UI controls and regeneration.

## `hw_viz.regeneration`

- `regenerate_scenarios()`: applies a single fitted terminal distribution while preserving the simulation structure (OU bridge)
- `regenerate_scenarios_surface()`: honours an edited distribution surface across horizons by rank-remapping each edited horizon's marginal

This is the strategy boundary for scenario regeneration; add further strategies here.

## `hw_viz.components.draggable_chart`

- A dependency-free Streamlit custom component (`index.html` + thin Python wrapper).
- Renders an SVG chart whose anchor points can be dragged; reports the dragged positions back to Python.
- All math (curve smoothing, distribution fitting) stays server-side: the component only reports where points were dragged, and the server sends authoritative anchor positions back on the next render.

## `hw_viz.exports`

- `short_rate_frame()`
- `yield_curve_frame()`
- `metadata_json()`

Keep export schemas stable. Downstream users are more likely to break on export changes than on UI changes.

## Extension Boundaries

Good next extensions:

- add sample datasets under `examples/`
- separate plotting helpers from `app.py` if the UI grows
- add alternative distribution fitting strategies behind a common interface
- add parameter-estimation workflows distinct from simulation controls

Avoid for now:

- mixing ad hoc notebook logic into package modules
- adding hidden state inside model functions
- coupling tests to Streamlit widget details
