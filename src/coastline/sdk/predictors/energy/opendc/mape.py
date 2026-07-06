"""MAPE calculator for comparing simulated vs actual power consumption."""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MapeComparator:
    """Compares simulated and actual power timeseries using MAPE metric."""

    def __init__(self, mape_window_minutes: int = 60) -> None:
        self.mape_window_minutes = mape_window_minutes
        self.mape_window = timedelta(minutes=mape_window_minutes)

    def compare(
        self,
        simulated_power: pd.DataFrame,
        actual_power: pd.DataFrame,
        simulation_end_time: datetime,
    ) -> dict[str, Any]:
        """Return MAPE (%) + window stats over the trailing mape_window_minutes.

        Both DataFrames must have [timestamp, power_draw] columns. Raises ValueError on empty/missing data.
        """
        for name, df in [("simulated", simulated_power), ("actual", actual_power)]:
            if df.empty:
                raise ValueError(f"Empty {name} power DataFrame")
            for col in ("timestamp", "power_draw"):
                if col not in df.columns:
                    raise ValueError(f"Missing '{col}' column in {name} DataFrame")

        window_end = simulation_end_time
        earliest_sim = pd.to_datetime(simulated_power["timestamp"].min())
        earliest_actual = pd.to_datetime(actual_power["timestamp"].min())
        earliest_available = max(earliest_sim, earliest_actual)

        ideal_start = window_end - self.mape_window
        window_start = max(ideal_start, earliest_available)

        sim_windowed = simulated_power[
            (pd.to_datetime(simulated_power["timestamp"]) >= window_start)
            & (pd.to_datetime(simulated_power["timestamp"]) <= window_end)
        ].copy()

        actual_windowed = actual_power[
            (pd.to_datetime(actual_power["timestamp"]) >= window_start)
            & (pd.to_datetime(actual_power["timestamp"]) <= window_end)
        ].copy()

        if sim_windowed.empty or actual_windowed.empty:
            raise ValueError(
                f"No data in MAPE window [{window_start} - {window_end}]: "
                f"{len(sim_windowed)} simulated, {len(actual_windowed)} actual"
            )

        aligned = self._align_timeseries(sim_windowed, actual_windowed)
        if aligned.empty:
            raise ValueError("Failed to align simulated and actual timeseries")

        simulated = np.array(aligned["simulated_power"].values)
        actual = np.array(aligned["actual_power"].values)

        mask = actual != 0
        if not np.any(mask):
            raise ValueError("All actual power values are zero, cannot calculate MAPE")

        percentage_errors = np.abs((actual[mask] - simulated[mask]) / actual[mask]) * 100
        mape = float(np.mean(percentage_errors))

        logger.info(
            "MAPE: %.2f%% (%d points, mean_sim=%.1fW, mean_actual=%.1fW)",
            mape,
            int(np.sum(mask)),
            float(np.mean(simulated)),
            float(np.mean(actual)),
        )

        return {
            "mape": mape,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "num_points": len(aligned),
            "mean_simulated": float(np.mean(simulated)),
            "mean_actual": float(np.mean(actual)),
        }

    @staticmethod
    def _align_timeseries(
        sim_df: pd.DataFrame,
        actual_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align timeseries by interpolation on a 60s grid."""
        sim = sim_df.copy()
        actual = actual_df.copy()
        sim["timestamp"] = pd.to_datetime(sim["timestamp"])
        actual["timestamp"] = pd.to_datetime(actual["timestamp"])

        sim = sim.rename(columns={"power_draw": "simulated_power"})
        actual = actual.rename(columns={"power_draw": "actual_power"})

        start = max(sim["timestamp"].min(), actual["timestamp"].min())
        end = min(sim["timestamp"].max(), actual["timestamp"].max())
        if start >= end:
            return pd.DataFrame({"timestamp": [], "simulated_power": [], "actual_power": []})

        grid = pd.date_range(start=start, end=end, freq="60s")
        if len(grid) == 0:
            return pd.DataFrame({"timestamp": [], "simulated_power": [], "actual_power": []})

        sim = sim.set_index("timestamp")
        sim_interp = sim.reindex(sim.index.union(grid)).interpolate(method="time")
        sim_interp = sim_interp.reindex(grid)

        actual = actual.set_index("timestamp")
        actual_interp = actual.reindex(actual.index.union(grid)).interpolate(method="time")
        actual_interp = actual_interp.reindex(grid)

        result = pd.DataFrame(
            {
                "timestamp": grid,
                "simulated_power": sim_interp["simulated_power"].values,
                "actual_power": actual_interp["actual_power"].values,
            }
        )
        return result.dropna()
