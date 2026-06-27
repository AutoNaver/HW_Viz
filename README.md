# Hull-White Interactive Rate Simulation

`hw-viz` is a local Streamlit application for exploring short-rate scenarios under the one-factor Hull-White model. It covers the current MVP workflow end to end:

- ingest a tenor/rate zero curve
- build an interpolated zero curve and discount factors
- reshape the starting curve by dragging tenor points directly (with optional smoothing)
- simulate short-rate and yield-curve scenarios
- edit the short-rate distribution by dragging quantile points, with edits spreading smoothly to adjacent horizons
- inspect a 3D surface across horizons, switchable between the short-rate distribution and the mean yield curve over time
- fit a smooth target distribution
- regenerate scenarios consistent with the edited distribution surface
- export regenerated scenarios and metadata

This README describes the implemented application, not the longer-term product vision.

## Current MVP Scope

The current implementation is intentionally narrow so it is stable and easy to extend:

- Input is treated as a zero-rate curve, not a full mixed-instrument bootstrap set.
- The UI is a single-page Streamlit app.
- Curve and distribution editing are drag-based, backed by a small self-contained Streamlit component.
- Curve drags propagate smoothly to neighbouring short tenors (smoothing can be switched off).
- Distribution editing drags quantile points (including the 1% / 99% tails); fitting uses a monotone spline quantile function.
- A distribution drag can spread to adjacent horizons (Gaussian in time, switchable, with an adjustable influence width).
- Regeneration honours the whole edited distribution surface via a rank-preserving (copula-preserving) marginal remap.
- Hull-White `a` and `sigma` are user inputs, not statistically estimated from market data.

## Repository Layout

```text
.
|-- app.py                  # Streamlit entrypoint
|-- hw_viz/
|   |-- curve.py            # Curve parsing, normalization, interpolation, drag smoothing
|   |-- hull_white.py       # Simulation config, theta/alpha, bond pricing, path generation
|   |-- distribution.py     # Quantile edits, spline-based fitting, sampling
|   |-- regeneration.py     # Horizon-conditioned scenario regeneration
|   |-- exports.py          # CSV/JSON export helpers
|   `-- components/         # Self-contained draggable-chart Streamlit component
|-- tests/
|   `-- test_hw_viz.py      # Quant and workflow tests
|-- docs/
|   |-- architecture.md     # System map and data flow
|   `-- development.md      # Local setup, testing, extension guidance
|-- examples/
|   `-- sample_curve.csv    # Example input file
`-- pyproject.toml
```

## Requirements

- Python 3.11+
- `streamlit`
- `numpy`
- `pandas`
- `scipy`
- `plotly`
- `pytest` for tests

Install in editable mode:

```bash
python -m pip install -e .
```

## Run The App

```bash
streamlit run app.py
```

The app starts with a default curve and supports either:

- uploading a CSV with `tenor` and `rate` columns
- editing the in-app curve table manually

Example CSV:

```csv
tenor,rate
1M,4.10
3M,4.05
6M,4.00
1Y,3.90
2Y,3.70
5Y,3.35
10Y,3.15
```

## Run Tests

```bash
pytest -q
```

The current test suite covers:

- tenor parsing and curve validation
- discount-factor monotonicity
- finite Hull-White drift construction
- seeded simulation reproducibility
- initial curve consistency of model-implied yields
- fitted distribution validity
- regenerated horizon quantiles
- export artifact generation

## Workflow

1. Provide a tenor/rate curve (upload, edit the table, or drag tenor points on the curve chart).
2. Choose `a`, `sigma`, horizon, step size, path count, and seed.
3. Run the initial Hull-White simulation.
4. Pick a horizon and drag the quantile points (`1%`, `5%`, `25%`, `50%`, `75%`, `95%`, `99%`) to reshape the short-tenor distribution, including its tails. Toggle "Spread edits to adjacent horizons" to flow an edit smoothly across time, and inspect the 3D distribution surface.
5. Regenerate scenarios from the edited distribution surface.
6. Download short-rate scenarios, yield-curve scenarios, and metadata.

## Assumptions And Limitations

- Rates entered above `1.0` are interpreted as percentages and converted internally.
- Day count is effectively `ACT/365`, with time steps expressed in year fractions.
- The default simulation is monthly and uses an Ornstein-Uhlenbeck state process.
- The short-rate central tendency `E[r(t)]` tracks the drawn zero curve `z(t)` (plus the usual variance term), rather than the textbook instantaneous forward `f(0, t) = z(t) + t z'(t)`. This keeps the simulated term structure faithful to the curve you draw — a steep edit no longer makes the forward overshoot and snap back — at the cost of exact arbitrage-free repricing for `t > 0`.
- Yield curves are produced from affine Hull-White bond-pricing formulas (which still use the market discount curve).
- Scenario regeneration conditions the process on the edited horizon distribution, then resumes standard dynamics after that horizon.
- The app is designed for local experimentation and demos, not production risk infrastructure.

## Maintenance Notes

- Keep quant logic in `hw_viz/` and UI wiring in `app.py`.
- Add tests before widening model scope.
- Prefer extending existing dataclasses and pure functions over adding UI-bound logic inside model modules.
- If you add new exported artifacts or controls, update both [docs/development.md](/c:/Users/naver/OneDrive/Desktop/HW_Viz/docs/development.md) and [docs/architecture.md](/c:/Users/naver/OneDrive/Desktop/HW_Viz/docs/architecture.md).

## Additional Docs

- [Architecture](/c:/Users/naver/OneDrive/Desktop/HW_Viz/docs/architecture.md)
- [Development Guide](/c:/Users/naver/OneDrive/Desktop/HW_Viz/docs/development.md)
- [Contributing](/c:/Users/naver/OneDrive/Desktop/HW_Viz/CONTRIBUTING.md)
