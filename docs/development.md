# Development Guide

## Local Setup

Create and activate a virtual environment, then install the package in editable mode:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the application:

```bash
streamlit run app.py
```

Run the tests:

```bash
pytest -q
```

## Input Contract

The application expects a tabular curve input with:

- `tenor`: string such as `1M`, `3M`, `1Y`, `10Y`
- `rate`: numeric value in decimal or percent form

Current parsing rules:

- `D`, `W`, `M`, `Y` tenor suffixes are supported
- values greater than `1.0` in absolute magnitude are treated as percentages
- input is interpreted as a zero-rate curve

## Key Public Types

## `SimulationConfig`

Defined in [hull_white.py](/c:/Users/naver/OneDrive/Desktop/HW_Viz/hw_viz/hull_white.py).

Fields:

- `mean_reversion_a`
- `volatility_sigma`
- `simulation_horizon_years`
- `time_step_months`
- `num_paths`
- `random_seed`

## `SimulationResult`

Primary in-memory output of both initial simulation and regeneration.

Fields:

- `time_grid`
- `maturity_grid`
- `short_rate_paths`
- `yield_curves`
- `alpha_curve`
- `theta_curve`
- `x_paths`

## `DistributionEdit`

Represents the edited terminal distribution:

- `horizon_time`
- `quantile_levels`
- `quantile_values`
- `left_tail_scale`
- `right_tail_scale`

When the distribution is edited through the draggable quantile editor, tail behaviour is expressed directly by the dragged 1% / 99% anchor values, so `left_tail_scale` / `right_tail_scale` are left at their neutral value of `1.0` and `quantile_values` holds the standard reporting quantiles interpolated from the anchors.

## Export Schemas

Short-rate export columns:

- `scenario_set`
- `path`
- `time_years`
- `short_rate`

Yield export columns:

- `scenario_set`
- `path`
- `time_years`
- `maturity_years`
- `yield_rate`

Metadata export contains:

- curve input rows
- simulation config
- edited distribution settings

## Testing Strategy

Add or update tests whenever you change:

- tenor parsing or rate normalization
- interpolation behavior
- simulation dynamics
- regeneration logic
- export schema

Prefer unit tests for numerical helpers and one end-to-end test for workflow continuity.

## Maintenance Rules

- Keep dataclasses as the primary public contract between modules.
- Favor deterministic randomness via explicit seeds.
- Do not let Streamlit session-state logic leak into `hw_viz/`.
- If the UI grows beyond a single file, move chart builders and section renderers into a `ui/` package without moving quant code.
- When introducing a new algorithm, write down its assumptions in the docstring and in the README limitations section.
