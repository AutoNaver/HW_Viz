from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hw_viz.distribution import DistributionEdit
from hw_viz.hull_white import SimulationConfig, SimulationResult


def short_rate_frame(result: SimulationResult, scenario_label: str) -> pd.DataFrame:
    path_ids = np.repeat(np.arange(result.short_rate_paths.shape[0]), result.time_grid.size)
    time_values = np.tile(result.time_grid, result.short_rate_paths.shape[0])
    rates = result.short_rate_paths.reshape(-1)
    return pd.DataFrame(
        {
            "scenario_set": scenario_label,
            "path": path_ids,
            "time_years": time_values,
            "short_rate": rates,
        }
    )


def yield_curve_frame(result: SimulationResult, scenario_label: str) -> pd.DataFrame:
    path_ids, time_ids, maturity_ids = np.indices(result.yield_curves.shape)
    return pd.DataFrame(
        {
            "scenario_set": scenario_label,
            "path": path_ids.reshape(-1),
            "time_years": result.time_grid[time_ids.reshape(-1)],
            "maturity_years": result.maturity_grid[maturity_ids.reshape(-1)],
            "yield_rate": result.yield_curves.reshape(-1),
        }
    )


def metadata_json(
    curve_frame: pd.DataFrame,
    config: SimulationConfig,
    edit: DistributionEdit,
    scenario_label: str,
) -> str:
    payload = {
        "scenario_set": scenario_label,
        "curve": curve_frame.to_dict(orient="records"),
        "config": {
            "mean_reversion_a": config.mean_reversion_a,
            "volatility_sigma": config.volatility_sigma,
            "simulation_horizon_years": config.simulation_horizon_years,
            "time_step_months": config.time_step_months,
            "num_paths": config.num_paths,
            "random_seed": config.random_seed,
        },
        "distribution_edit": {
            "horizon_time": edit.horizon_time,
            "quantile_levels": edit.quantile_levels.tolist(),
            "quantile_values": edit.quantile_values.tolist(),
            "left_tail_scale": edit.left_tail_scale,
            "right_tail_scale": edit.right_tail_scale,
        },
    }
    return json.dumps(payload, indent=2)
