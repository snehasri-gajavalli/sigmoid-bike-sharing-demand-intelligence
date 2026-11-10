"""Assumption-explicit fleet sizing and station-level redistribution helpers.

The supplied UCI bike-sharing files contain citywide completed rentals only.
They do not contain station inventory, trip duration, failed requests or bike flows.
Therefore citywide fleet outputs are scenario estimates, not verified requirements.
Station transfer planning is available only when the caller supplies station-level data.
"""

from __future__ import annotations

import math
from itertools import product
from numbers import Real

import numpy as np
import pandas as pd


def _finite_number(name: str, value: float) -> float:
    """Convert a numeric input to float and reject NaN/infinity."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric value.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value.")
    return value


def estimate_fleet(
    hourly_rentals: float,
    trip_minutes: float = 20.0,
    spatial_buffer: float = 2.0,
    maintenance_share: float = 0.10,
) -> dict:
    """Estimate an assumption-based fleet scenario using Little's Law: L = lambda * W."""
    hourly_rentals = _finite_number("hourly_rentals", hourly_rentals)
    trip_minutes = _finite_number("trip_minutes", trip_minutes)
    spatial_buffer = _finite_number("spatial_buffer", spatial_buffer)
    maintenance_share = _finite_number("maintenance_share", maintenance_share)

    if hourly_rentals < 0:
        raise ValueError("hourly_rentals cannot be negative.")
    if not 5 <= trip_minutes <= 120:
        raise ValueError("trip_minutes must be between 5 and 120.")
    if not 1 <= spatial_buffer <= 10:
        raise ValueError("spatial_buffer must be between 1 and 10.")
    if not 0 <= maintenance_share < 0.9:
        raise ValueError("maintenance_share must be in [0, 0.9).")

    bikes_in_motion = hourly_rentals * (trip_minutes / 60.0)
    bikes_positioned = bikes_in_motion * spatial_buffer
    total_fleet = math.ceil(bikes_positioned / (1.0 - maintenance_share))

    return {
        "hourly_rentals": hourly_rentals,
        "trip_minutes": trip_minutes,
        "spatial_buffer": spatial_buffer,
        "maintenance_share": maintenance_share,
        "bikes_in_motion": bikes_in_motion,
        "bikes_positioned": bikes_positioned,
        "estimated_fleet": total_fleet,
        "formula": (
            "ceil(hourly_rentals × trip_minutes/60 × spatial_buffer "
            "÷ (1-maintenance_share))"
        ),
        "classification": (
            "ASSUMPTION-BASED SCENARIO — not a measured fleet requirement"
        ),
    }


def observed_demand_percentiles(hour_df: pd.DataFrame, year: int = 2012) -> dict:
    """Return observed completed-rental percentiles for one UCI dataset year."""
    required = {"cnt", "yr"}
    missing = required - set(hour_df.columns)
    if missing:
        raise ValueError(f"hour_df is missing columns: {sorted(missing)}")

    if isinstance(year, bool) or not isinstance(year, (int, np.integer)):
        raise ValueError("year must be an integer, such as 2011 or 2012.")

    year_code = int(year) - 2011
    if year_code not in (0, 1):
        raise ValueError("year must be 2011 or 2012 for this dataset.")

    values = pd.to_numeric(
        hour_df.loc[hour_df["yr"] == year_code, "cnt"],
        errors="coerce",
    ).dropna()

    if values.empty:
        raise ValueError(f"No hourly observations found for {year}.")

    return {
        "year": int(year),
        "n_hours": int(values.size),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "maximum": float(values.max()),
    }


def fleet_sensitivity(
    hourly_rentals: float,
    trip_minutes_values=(10, 20, 30),
    spatial_buffer_values=(1.5, 2.0, 3.0),
    maintenance_values=(0.05, 0.10, 0.15),
) -> pd.DataFrame:
    """Generate a grid of fleet scenarios; all inputs are explicit assumptions."""
    rows = []
    for trip, buffer, maintenance in product(
        trip_minutes_values, spatial_buffer_values, maintenance_values
    ):
        result = estimate_fleet(
            hourly_rentals, trip, buffer, maintenance
        )
        rows.append({
            "Hourly rentals": result["hourly_rentals"],
            "Trip duration (min)": result["trip_minutes"],
            "Spatial buffer": result["spatial_buffer"],
            "Maintenance share": result["maintenance_share"],
            "Bikes in motion (Little's Law)": result["bikes_in_motion"],
            "Scenario fleet": result["estimated_fleet"],
        })
    return pd.DataFrame(rows)


def demand_timing_recommendations(hour_df: pd.DataFrame) -> dict:
    """Summarise citywide demand timing without identifying station shortages."""
    required = {"hr", "workingday", "cnt"}
    missing = required - set(hour_df.columns)
    if missing:
        raise ValueError(f"hour_df is missing columns: {sorted(missing)}")

    data = hour_df.copy()
    for column in ("hr", "workingday", "cnt"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["hr", "workingday", "cnt"])
    if not data["hr"].between(0, 23).all():
        raise ValueError("hr values must be between 0 and 23.")
    if not data["workingday"].isin([0, 1]).all():
        raise ValueError("workingday values must be 0 or 1.")
    if (data["cnt"] < 0).any():
        raise ValueError("cnt cannot contain negative values.")

    profile = (
        data.groupby(["workingday", "hr"])["cnt"]
        .mean()
        .rename("mean_completed_rentals")
        .reset_index()
    )
    working = profile[profile["workingday"] == 1].nlargest(
        4, "mean_completed_rentals"
    )
    non_working = profile[profile["workingday"] == 0].nlargest(
        4, "mean_completed_rentals"
    )

    def records(frame):
        return [
            {
                "hour": int(row.hr),
                "mean_completed_rentals": round(
                    float(row.mean_completed_rentals), 1
                ),
            }
            for row in frame.itertuples(index=False)
        ]

    return {
        "working_day_top_hours": records(working),
        "non_working_day_top_hours": records(non_working),
        "working_day_trial_windows": [
            (
                "Proposed trial: complete a morning positioning round before "
                "the observed morning peak (test a finish target around 07:00)."
            ),
            (
                "Proposed trial: complete an afternoon positioning round before "
                "the observed evening peak (test a finish target around 16:30)."
            ),
        ],
        "non_working_day_trial_guidance": (
            "Use the measured non-working-day hourly profile to schedule a separate "
            "midday-oriented trial; do not automatically reuse the working-day "
            "commute schedule."
        ),
        "limitation": (
            "These are citywide timing signals only. They do not identify which "
            "station has a surplus or shortage. The suggested positioning times "
            "are trial hypotheses, not outputs calculated from station-level flows."
        ),
    }


def plan_station_transfers(
    station_snapshot: pd.DataFrame,
    forecast_horizon_hours: float = 2.0,
    reserve_multiplier: float = 1.15,
    max_transfer_per_station: int | None = None,
) -> pd.DataFrame:
    """Create a heuristic station-level donor/receiver shortlist.

    Required columns:
      station_id, available_bikes, forecast_rentals_per_hour

    target_stock = forecast rentals over the planning horizon × reserve multiplier.

    This heuristic does not optimise donor-to-receiver pairings or routes. It ignores
    predicted returns, dock capacity, travel times, vehicle capacity, route costs
    and simultaneous flows. Validate the output operationally before dispatch.
    """
    if not isinstance(station_snapshot, pd.DataFrame):
        raise ValueError("station_snapshot must be a pandas DataFrame.")

    required_columns = [
        "station_id",
        "available_bikes",
        "forecast_rentals_per_hour",
    ]
    missing = set(required_columns) - set(station_snapshot.columns)
    if missing:
        raise ValueError(
            f"station_snapshot is missing columns: {sorted(missing)}"
        )

    forecast_horizon_hours = _finite_number(
        "forecast_horizon_hours", forecast_horizon_hours
    )
    reserve_multiplier = _finite_number(
        "reserve_multiplier", reserve_multiplier
    )

    if forecast_horizon_hours <= 0:
        raise ValueError("forecast_horizon_hours must be positive.")
    if reserve_multiplier < 1:
        raise ValueError("reserve_multiplier must be at least 1.")

    if max_transfer_per_station is not None:
        if (
            isinstance(max_transfer_per_station, bool)
            or not isinstance(max_transfer_per_station, (int, np.integer))
            or max_transfer_per_station < 0
        ):
            raise ValueError(
                "max_transfer_per_station must be a non-negative integer or None."
            )

    result = station_snapshot[required_columns].copy()

    if result["station_id"].isna().any():
        raise ValueError("station_id cannot contain missing values.")
    if result["station_id"].astype(str).str.strip().eq("").any():
        raise ValueError("station_id cannot contain empty values.")
    if result["station_id"].duplicated().any():
        raise ValueError("station_id values must be unique.")

    for column in ("available_bikes", "forecast_rentals_per_hour"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"{column} must contain finite numeric values.")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite numeric values.")
        if (result[column] < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")

    result["target_stock"] = (
        result["forecast_rentals_per_hour"]
        * forecast_horizon_hours
        * reserve_multiplier
    ).apply(math.ceil)

    result["surplus_bikes"] = (
        result["available_bikes"] - result["target_stock"]
    ).clip(lower=0).astype(int)
    result["deficit_bikes"] = (
        result["target_stock"] - result["available_bikes"]
    ).clip(lower=0).astype(int)

    if max_transfer_per_station is not None:
        result["surplus_bikes"] = result["surplus_bikes"].clip(
            upper=int(max_transfer_per_station)
        )
        result["deficit_bikes"] = result["deficit_bikes"].clip(
            upper=int(max_transfer_per_station)
        )

    result["action"] = "Monitor"
    result.loc[
        result["surplus_bikes"] > 0, "action"
    ] = "Potential donor station"
    result.loc[
        result["deficit_bikes"] > 0, "action"
    ] = "Potential receiver station"

    result["transfer_priority"] = result[
        ["surplus_bikes", "deficit_bikes"]
    ].max(axis=1)

    return (
        result.sort_values(
            "transfer_priority", ascending=False, kind="stable"
        )
        .reset_index(drop=True)
    )
