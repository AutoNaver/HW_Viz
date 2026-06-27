from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import PchipInterpolator

from hw_viz.components.draggable_chart import draggable_chart
from hw_viz.curve import apply_curve_drag, build_curve, default_curve_frame, parse_tenor
from hw_viz.distribution import (
    DISTRIBUTION_ANCHOR_LEVELS,
    QUANTILE_LEVELS,
    build_anchor_surface,
    build_distribution_edit_from_anchors,
    empirical_quantiles,
    fit_distribution_from_quantiles,
    propagate_horizon_drag,
)
from hw_viz.exports import metadata_json, short_rate_frame, yield_curve_frame
from hw_viz.hull_white import SimulationConfig, simulate_hull_white_paths
from hw_viz.regeneration import regenerate_scenarios_surface


st.set_page_config(page_title="Hull-White Interactive Rate Simulation", layout="wide")


def download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def parse_uploaded_curve(uploaded_file) -> pd.DataFrame:
    content = StringIO(uploaded_file.getvalue().decode("utf-8"))
    return pd.read_csv(content)


def short_rate_path_figure(time_grid: np.ndarray, paths: np.ndarray, title: str) -> go.Figure:
    figure = go.Figure()
    sample_count = min(15, paths.shape[0])
    for path in paths[:sample_count]:
        figure.add_trace(go.Scatter(x=time_grid, y=path, mode="lines", line={"width": 1}, opacity=0.45, showlegend=False))
    figure.update_layout(title=title, xaxis_title="Time (years)", yaxis_title="Short rate")
    return figure


def distribution_figure(samples: np.ndarray, fitted_x: np.ndarray | None = None, fitted_pdf: np.ndarray | None = None, title: str = "") -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Histogram(x=samples, histnorm="probability density", nbinsx=40, opacity=0.55, name="Empirical"))
    if fitted_x is not None and fitted_pdf is not None:
        figure.add_trace(go.Scatter(x=fitted_x, y=fitted_pdf, mode="lines", line={"width": 3}, name="Fitted"))
    figure.update_layout(barmode="overlay", title=title, xaxis_title="Short rate", yaxis_title="Density")
    return figure


def distribution_surface_figure(
    times: np.ndarray,
    anchor_levels: np.ndarray,
    surface: np.ndarray,
    max_horizons: int = 24,
    value_points: int = 90,
    title: str = "",
) -> go.Figure:
    """3D density surface of the short-rate distribution across horizons."""
    indices = np.arange(1, len(times))
    if indices.size > max_horizons:
        indices = np.unique(np.linspace(1, len(times) - 1, max_horizons).round().astype(int))
    selected_times = times[indices]
    fits = [fit_distribution_from_quantiles(anchor_levels, surface[i]) for i in indices]
    low = min(fit.value_grid[0] for fit in fits)
    high = max(fit.value_grid[-1] for fit in fits)
    value_grid = np.linspace(low, high, value_points)
    density = np.empty((value_points, indices.size), dtype=float)
    for column, fit in enumerate(fits):
        density[:, column] = fit.pdf(value_grid)
    figure = go.Figure(data=[go.Surface(x=selected_times, y=value_grid, z=density, colorscale="Viridis", showscale=False)])
    figure.update_layout(
        title=title,
        scene={
            "xaxis_title": "Horizon (years)",
            "yaxis_title": "Short rate",
            "zaxis_title": "Density",
            "camera": {"eye": {"x": 1.7, "y": -1.5, "z": 0.9}},
        },
        height=600,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return figure


def mean_yield_surface_figure(result, title: str = "") -> go.Figure:
    """3D surface of the mean yield curve (averaged over paths) evolving over time."""
    mean_yields = result.yield_curves.mean(axis=0)  # (num_times, num_maturities)
    figure = go.Figure(
        data=[
            go.Surface(
                x=result.time_grid,
                y=result.maturity_grid,
                z=mean_yields.T,
                colorscale="Cividis",
                showscale=False,
            )
        ]
    )
    figure.update_layout(
        title=title,
        scene={
            "xaxis_title": "Time (years)",
            "yaxis_title": "Maturity (years)",
            "zaxis_title": "Yield",
            "camera": {"eye": {"x": 1.7, "y": -1.5, "z": 0.9}},
        },
        height=600,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return figure


def yield_snapshot_figure(result, title: str) -> go.Figure:
    figure = go.Figure()
    snapshot_indices = np.unique(
        np.array(
            [
                0,
                min(len(result.time_grid) - 1, len(result.time_grid) // 4),
                min(len(result.time_grid) - 1, len(result.time_grid) // 2),
                len(result.time_grid) - 1,
            ]
        )
    )
    mean_yields = result.yield_curves.mean(axis=0)
    for index in snapshot_indices:
        figure.add_trace(
            go.Scatter(
                x=result.maturity_grid,
                y=mean_yields[index],
                mode="lines+markers",
                name=f"t={result.time_grid[index]:.2f}",
            )
        )
    figure.update_layout(title=title, xaxis_title="Maturity (years)", yaxis_title="Yield")
    return figure


def init_state() -> None:
    st.session_state.setdefault("curve_frame", default_curve_frame())
    st.session_state.setdefault("initial_result", None)
    st.session_state.setdefault("regenerated_result", None)
    st.session_state.setdefault("fitted_distribution", None)
    st.session_state.setdefault("distribution_edit", None)
    st.session_state.setdefault("curve", None)
    st.session_state.setdefault("config", None)
    st.session_state.setdefault("curve_editor_version", 0)
    st.session_state.setdefault("curve_drag_event", 0)
    st.session_state.setdefault("dist_surface", None)
    st.session_state.setdefault("dist_base_surface", None)
    st.session_state.setdefault("dist_drag_event", 0)


init_state()

st.title("Hull-White Interactive Rate Simulation")
st.caption("Streamlit MVP for curve ingestion, Hull-White simulation, quantile-based distribution editing, and scenario regeneration.")

with st.sidebar:
    st.header("Model Controls")
    mean_reversion_a = st.number_input("Mean reversion a", min_value=0.01, max_value=2.0, value=0.12, step=0.01, format="%.4f")
    volatility_sigma = st.number_input("Volatility sigma", min_value=0.0, max_value=0.2, value=0.01, step=0.001, format="%.4f")
    simulation_horizon_years = st.number_input("Simulation horizon (years)", min_value=1.0, max_value=30.0, value=10.0, step=1.0)
    time_step_months = st.selectbox("Time step (months)", options=[1, 3, 6], index=0)
    num_paths = st.slider("Number of paths", min_value=100, max_value=5000, value=1000, step=100)
    random_seed = st.number_input("Random seed", min_value=0, max_value=100000, value=7, step=1)

config = SimulationConfig(
    mean_reversion_a=float(mean_reversion_a),
    volatility_sigma=float(volatility_sigma),
    simulation_horizon_years=float(simulation_horizon_years),
    time_step_months=int(time_step_months),
    num_paths=int(num_paths),
    random_seed=int(random_seed),
)

def curve_tenor_points(frame: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (tenors, maturities, rates) sorted by maturity for the drag chart."""
    working = frame.loc[:, ["tenor", "rate"]].copy()
    working["tenor"] = working["tenor"].astype(str).str.strip()
    working["maturity"] = working["tenor"].map(parse_tenor)
    working["rate"] = pd.to_numeric(working["rate"], errors="raise").astype(float)
    working = working.sort_values("maturity", kind="stable").reset_index(drop=True)
    return working["tenor"].tolist(), working["maturity"].to_numpy(float), working["rate"].to_numpy(float)


st.subheader("1. Curve Input")
st.caption("Drag any tenor point up or down to reshape the starting rate curve. Short neighbours follow to keep it smooth; the long end stays put.")

upload_col, option_col = st.columns([1.0, 1.0])
with upload_col:
    uploaded_file = st.file_uploader("Upload tenor/rate CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            st.session_state["curve_frame"] = parse_uploaded_curve(uploaded_file)
            st.session_state["curve_editor_version"] += 1
        except Exception as exc:
            st.error(f"Could not parse uploaded CSV: {exc}")
with option_col:
    smooth_curve = st.checkbox("Smooth neighbouring tenors when dragging", value=True)

curve_chart_col, curve_table_col = st.columns([1.4, 1.0])
with curve_chart_col:
    try:
        tenors, maturities, rates = curve_tenor_points(st.session_state["curve_frame"])
        smooth_x = np.geomspace(maturities[0], maturities[-1], 120)
        smooth_y = PchipInterpolator(maturities, rates, extrapolate=True)(smooth_x)
        rate_pad = max(0.25, 0.2 * (rates.max() - rates.min()))
        curve_spec = {
            "title": "Initial rate curve",
            "xLabel": "Tenor",
            "yLabel": "Rate (%)",
            "xType": "log",
            "dragAxis": "y",
            "xRange": [float(maturities[0] * 0.85), float(maturities[-1] * 1.15)],
            "yRange": [float(rates.min() - rate_pad), float(rates.max() + rate_pad)],
            "xTicks": [float(m) for m in maturities],
            "xTickLabels": list(tenors),
            "background": [
                {"x": [float(v) for v in smooth_x], "y": [float(v) for v in smooth_y], "color": "#c5c9cf", "width": 2}
            ],
            "anchors": [
                {"id": tenor, "x": float(m), "y": float(r), "label": tenor}
                for tenor, m, r in zip(tenors, maturities, rates)
            ],
            "hint": "drag a point",
            "height": 340,
        }
        drag_result = draggable_chart(curve_spec, key="curve_drag")
        if drag_result and int(drag_result.get("event", 0)) != st.session_state["curve_drag_event"]:
            st.session_state["curve_drag_event"] = int(drag_result["event"])
            dragged_id = drag_result["lastId"]
            new_value = next(a["y"] for a in drag_result["anchors"] if a["id"] == dragged_id)
            index = tenors.index(dragged_id)
            updated_rates = apply_curve_drag(maturities, rates, index, float(new_value), smoothing=smooth_curve)
            new_frame = pd.DataFrame({"tenor": tenors, "rate": np.round(updated_rates, 6)})
            st.session_state["curve_frame"] = new_frame
            st.session_state["curve_editor_version"] += 1
            st.rerun()
    except Exception as exc:
        st.warning(f"Curve validation pending: {exc}")

with curve_table_col:
    with st.expander("Edit curve as a table", expanded=False):
        edited_curve = st.data_editor(
            st.session_state["curve_frame"],
            num_rows="dynamic",
            use_container_width=True,
            key=f"curve_editor_{st.session_state['curve_editor_version']}",
        )
        st.session_state["curve_frame"] = edited_curve

    try:
        preview_curve = build_curve(st.session_state["curve_frame"])
        preview_df = pd.DataFrame(
            {
                "maturity_years": preview_curve.maturities,
                "zero_rate": preview_curve.zero_rates,
                "discount_factor": preview_curve.discount_factors,
            }
        )
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Curve validation pending: {exc}")

run_initial = st.button("Run Initial Simulation", type="primary", use_container_width=True)

if run_initial:
    try:
        curve = build_curve(st.session_state["curve_frame"])
        initial_result = simulate_hull_white_paths(curve, config)
        st.session_state["curve"] = curve
        st.session_state["config"] = config
        st.session_state["initial_result"] = initial_result
        st.session_state["regenerated_result"] = None
        st.session_state["dist_surface"] = None
        st.session_state["dist_base_surface"] = None
        st.success("Initial Hull-White simulation complete.")
    except Exception as exc:
        st.error(f"Initial simulation failed: {exc}")

initial_result = st.session_state["initial_result"]
curve = st.session_state["curve"]

if initial_result is not None and curve is not None:
    st.subheader("2. Initial Simulation Outputs")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(short_rate_path_figure(initial_result.time_grid, initial_result.short_rate_paths, "Sample short-rate paths"), use_container_width=True)
    with right:
        st.plotly_chart(yield_snapshot_figure(initial_result, "Mean yield curve snapshots"), use_container_width=True)

    st.subheader("3. Distribution Editing")
    st.caption("Drag the quantile points left/right to reshape the short-tenor horizon distribution. The outer points (1% / 99%) set the tails. With spreading enabled, an edit flows smoothly into adjacent horizons.")

    time_grid = initial_result.time_grid

    # The full distribution surface: a row of quantile anchors per time index.
    base_surface = build_anchor_surface(initial_result.short_rate_paths, DISTRIBUTION_ANCHOR_LEVELS)
    if (
        st.session_state["dist_surface"] is None
        or np.asarray(st.session_state["dist_surface"]).shape != base_surface.shape
    ):
        st.session_state["dist_surface"] = base_surface.copy()
        st.session_state["dist_base_surface"] = base_surface.copy()

    horizon_time = st.slider(
        "Selected horizon (years)",
        min_value=float(time_grid[1]),
        max_value=float(time_grid[-1]),
        value=float(min(5.0, time_grid[-1])),
        step=float(config.dt),
    )
    horizon_index = int(np.argmin(np.abs(time_grid - horizon_time)))
    horizon_value = float(time_grid[horizon_index])

    spread_col, width_col, reset_col = st.columns([1.4, 1.4, 1.0])
    with spread_col:
        spread_edits = st.checkbox("Spread edits to adjacent horizons", value=True)
    with width_col:
        influence_years = st.slider(
            "Horizon influence (years)",
            min_value=0.25,
            max_value=float(max(1.0, time_grid[-1] / 2.0)),
            value=float(min(1.0, max(0.5, time_grid[-1] / 6.0))),
            step=0.25,
            disabled=not spread_edits,
        )
    with reset_col:
        if st.button("Reset distribution", use_container_width=True):
            st.session_state["dist_surface"] = base_surface.copy()
            st.rerun()

    surface = np.asarray(st.session_state["dist_surface"], dtype=float)
    dist_values = surface[horizon_index]
    base_samples = initial_result.short_rate_paths[:, horizon_index]
    base_quantiles = empirical_quantiles(base_samples, QUANTILE_LEVELS)

    # Draggable CDF editor for the selected horizon: x = short rate, y = cumulative probability.
    value_span = max(dist_values[-1] - dist_values[0], 1e-4)
    x_lo = float(min(base_samples.min(), dist_values[0]) - 0.8 * value_span)
    x_hi = float(max(base_samples.max(), dist_values[-1]) + 0.8 * value_span)
    sorted_samples = np.sort(base_samples)
    empirical_cdf = np.linspace(0.0, 1.0, sorted_samples.size)
    in_range = (sorted_samples >= x_lo) & (sorted_samples <= x_hi)
    ref_x = sorted_samples[in_range][:: max(1, sorted_samples.size // 80)]
    ref_y = empirical_cdf[in_range][:: max(1, sorted_samples.size // 80)]

    dist_spec = {
        "title": f"Horizon distribution (t = {horizon_value:.2f}y)",
        "xLabel": "Short rate",
        "yLabel": "Cumulative probability",
        "dragAxis": "x",
        "monotonicX": True,
        "xRange": [x_lo, x_hi],
        "yRange": [0.0, 1.0],
        "yTicks": [0.0, 0.25, 0.5, 0.75, 1.0],
        "background": [
            {"x": [float(v) for v in ref_x], "y": [float(v) for v in ref_y], "color": "#c5c9cf", "width": 2, "dash": "6,4"}
        ],
        "anchors": [
            {"id": f"q{int(level * 100)}", "x": float(value), "y": float(level), "label": f"{int(level * 100)}%"}
            for level, value in zip(DISTRIBUTION_ANCHOR_LEVELS, dist_values)
        ],
        "hint": "drag left / right",
        "height": 360,
    }
    dist_result = draggable_chart(dist_spec, key="dist_drag")
    if dist_result and int(dist_result.get("event", 0)) != st.session_state["dist_drag_event"]:
        st.session_state["dist_drag_event"] = int(dist_result["event"])
        dragged_id = dist_result["lastId"]
        level_index = next(
            i for i, level in enumerate(DISTRIBUTION_ANCHOR_LEVELS) if f"q{int(level * 100)}" == dragged_id
        )
        new_value = next(a["x"] for a in dist_result["anchors"] if a["id"] == dragged_id)
        st.session_state["dist_surface"] = propagate_horizon_drag(
            times=time_grid,
            surface=surface,
            time_index=horizon_index,
            level_index=level_index,
            new_value=float(new_value),
            smoothing=spread_edits,
            length_scale=float(influence_years),
        )
        st.rerun()

    edit = build_distribution_edit_from_anchors(DISTRIBUTION_ANCHOR_LEVELS, dist_values, horizon_value)
    fitted = fit_distribution_from_quantiles(DISTRIBUTION_ANCHOR_LEVELS, dist_values)
    edited_mask = np.any(np.abs(surface - np.asarray(st.session_state["dist_base_surface"])) > 1e-6, axis=1)
    st.session_state["distribution_edit"] = edit
    st.session_state["fitted_distribution"] = fitted

    dist_left, dist_right = st.columns(2)
    with dist_left:
        st.plotly_chart(
            distribution_figure(base_samples, fitted.value_grid, fitted.pdf_grid, "Selected horizon: empirical vs fitted target"),
            use_container_width=True,
        )
    with dist_right:
        comparison_df = pd.DataFrame(
            {
                "quantile": [f"{int(level * 100)}%" for level in QUANTILE_LEVELS],
                "original": base_quantiles,
                "edited_target": edit.quantile_values,
            }
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        edited_count = int(edited_mask.sum())
        st.write(f"Edited horizons: **{edited_count}** of {time_grid.size - 1}. Dragging defines a monotone spline quantile function per horizon; regeneration remaps each edited horizon's marginal while preserving path structure.")

    view_3d = st.radio(
        "3D view",
        ["Short-rate distribution", "Mean yield curve", "Hide"],
        horizontal=True,
        index=0,
    )
    if view_3d == "Short-rate distribution":
        st.plotly_chart(
            distribution_surface_figure(time_grid, DISTRIBUTION_ANCHOR_LEVELS, surface, title="Short-rate density across horizons"),
            use_container_width=True,
        )
    elif view_3d == "Mean yield curve":
        st.plotly_chart(
            mean_yield_surface_figure(initial_result, title="Mean yield curve over time (initial simulation)"),
            use_container_width=True,
        )

    st.subheader("4. Regenerated Scenarios and Export")
    regenerate = st.button("Regenerate Scenarios", use_container_width=True)
    if regenerate:
        regenerated_result = regenerate_scenarios_surface(
            curve, config, initial_result, surface, edited_mask, DISTRIBUTION_ANCHOR_LEVELS
        )
        st.session_state["regenerated_result"] = regenerated_result
        st.success("Regenerated scenario set is ready.")

regenerated_result = st.session_state["regenerated_result"]
fitted = st.session_state["fitted_distribution"]
edit = st.session_state["distribution_edit"]

if regenerated_result is not None and fitted is not None and edit is not None:
    regen_left, regen_right = st.columns(2)
    with regen_left:
        st.plotly_chart(short_rate_path_figure(regenerated_result.time_grid, regenerated_result.short_rate_paths, "Regenerated short-rate paths"), use_container_width=True)
    with regen_right:
        st.plotly_chart(yield_snapshot_figure(regenerated_result, "Regenerated mean yield snapshots"), use_container_width=True)

    horizon_index = int(np.argmin(np.abs(regenerated_result.time_grid - edit.horizon_time)))
    regenerated_samples = regenerated_result.short_rate_paths[:, horizon_index]
    comparison_figure = go.Figure()
    comparison_figure.add_trace(go.Histogram(x=regenerated_samples, histnorm="probability density", opacity=0.55, name="Regenerated"))
    comparison_figure.add_trace(go.Scatter(x=fitted.value_grid, y=fitted.pdf_grid, mode="lines", line={"width": 3}, name="Target fit"))
    comparison_figure.update_layout(title="Regenerated horizon distribution", xaxis_title="Short rate", yaxis_title="Density")
    st.plotly_chart(comparison_figure, use_container_width=True)

    short_rate_export = short_rate_frame(regenerated_result, "regenerated")
    yield_export = yield_curve_frame(regenerated_result, "regenerated")
    metadata_export = metadata_json(st.session_state["curve_frame"], st.session_state["config"], edit, "regenerated")

    export_left, export_center, export_right = st.columns(3)
    with export_left:
        st.download_button(
            "Download short-rate scenarios",
            data=download_csv(short_rate_export),
            file_name="short_rate_scenarios.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_center:
        st.download_button(
            "Download yield-curve scenarios",
            data=download_csv(yield_export),
            file_name="yield_curve_scenarios.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_right:
        st.download_button(
            "Download metadata",
            data=metadata_export.encode("utf-8"),
            file_name="scenario_metadata.json",
            mime="application/json",
            use_container_width=True,
        )
