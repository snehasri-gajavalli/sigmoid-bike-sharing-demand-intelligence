"""
Gemini-backed natural-language assistant for the Sigmoid bike-sharing project.

Safety design:
- Gemini only selects from an explicit, allow-listed set of analytical tools.
- Function arguments are validated locally before execution.
- Functions run on local Pandas data / the already-loaded prediction model.
- The model receives aggregate tool results, not the raw CSV or executable Python.
- If no API key is configured or the API fails, deterministic keyword routing remains available.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import date
from typing import Any, Callable

import pandas as pd

from fleet_planning import (
    demand_timing_recommendations,
    estimate_fleet,
    observed_demand_percentiles,
)

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
MAX_QUESTION_LENGTH = 2000

SYSTEM_INSTRUCTION = """
You are the Bike Sharing Demand Intelligence assistant for an operations manager.

Your task is to understand the user's question and select the most appropriate
one of the supplied project-analysis functions. You MUST call a function.
Do not answer from general knowledge and do not invent or estimate missing
operational facts.

Tool selection guidance:
- Hourly profile: busiest / quietest hours, demand at a given day type.
- Day-type comparison: working days vs non-working days / weekends.
- Customer segments: casual vs registered rental patterns.
- Weather summary: weather and completed-rental associations.
- Fleet scenario: assumption-based fleet calculation. Use the observed 2012 P95
  unless the user explicitly requests P90, P99, maximum or a custom demand rate.
  Defaults, only when not stated by the user: 20-minute trips, 2.0 spatial
  positioning buffer, 10% maintenance share. If custom demand is not used,
  set custom_hourly_rentals to 0.
- Redistribution recommendation: whether to investigate redistribution or buy
  bikes, timing guidance, pilot proposal and evidence limitations.
- Demand prediction: call only when the user has supplied all required date,
  hour, weather code and four normalized weather inputs. Valid dataset dates
  are 2011-01-01 through 2012-12-31. Do not invent missing prediction inputs.
- Data limitations: use for station-specific stock-outs, routes, trip duration,
  failed rentals, live availability, current operations, or a question the
  supplied data cannot answer.

Important evidence rules:
- `cnt` is completed rentals, not unmet demand or true latent demand.
- The source data is citywide. It does not identify station shortages, surplus
  stations, trip durations, failed rental attempts or station flows.
- Fleet outputs are scenarios based on explicit assumptions, not verified
  fleet requirements.
- Weather results are observational associations, not causal effects.
- Weather category 4 has very few records; mention this caveat when relevant.
- Do not choose a station or invent operational counts.
- Do not create multiple independent claims if one function can answer the
  question. If the request clearly asks two distinct supported things, you may
  request up to two functions.
"""

FINAL_ANSWER_INSTRUCTION = """
Write a concise, practical answer for a bike-sharing operations manager using
ONLY the function result(s) in the conversation. State whether each number is
observed, predicted or an assumption-based scenario. Explain the business
meaning, but do not add facts, numeric values, dates, thresholds or operational
claims not present in the function result. Do not claim station-level shortages
or causal effects from citywide completed-rental data. If the function result
says the data cannot answer the question, explain which operational data is
needed. Avoid numbered lists and avoid adding new numeric examples.
"""

# JSON schemas exposed to Gemini. Dispatch is independently allow-listed below.
FUNCTION_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_hourly_demand_profile",
        "description": "Get the observed average hourly completed-rental profile and busiest hours. Use when the user asks when demand is highest or asks about a particular day type.",
        "parameters": {
            "type": "object",
            "properties": {
                "day_type": {
                    "type": "string",
                    "enum": ["all_days", "working_days", "non_working_days"],
                    "description": "Which days to analyse."
                }
            },
            "required": ["day_type"],
        },
    },
    {
        "name": "compare_day_types",
        "description": "Compare observed average completed rentals and commute-window shares between working and non-working days.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_customer_segment_summary",
        "description": "Compare total rentals and casual/registered customer shares for working and non-working days.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_weather_demand_summary",
        "description": "Summarise observed completed rentals by weather category, including count and change versus clear weather.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "calculate_fleet_scenario",
        "description": "Calculate an assumption-based fleet scenario using Little's Law and the observed 2012 demand percentile or a custom hourly demand rate.",
        "parameters": {
            "type": "object",
            "properties": {
                "demand_basis": {
                    "type": "string",
                    "enum": ["observed_p90_2012", "observed_p95_2012", "observed_p99_2012", "observed_max_2012", "custom"],
                    "description": "Demand reference to use."
                },
                "custom_hourly_rentals": {
                    "type": "number",
                    "description": "Custom completed rentals per hour; set to 0 unless demand_basis is custom."
                },
                "trip_minutes": {
                    "type": "number",
                    "description": "Assumed average trip duration in minutes (5–120)."
                },
                "spatial_buffer": {
                    "type": "number",
                    "description": "Assumed spatial positioning buffer (1–10)."
                },
                "maintenance_share": {
                    "type": "number",
                    "description": "Assumed share unavailable for maintenance, from 0 to less than 0.9."
                },
            },
            "required": ["demand_basis", "custom_hourly_rentals", "trip_minutes", "spatial_buffer", "maintenance_share"],
        },
    },
    {
        "name": "get_redistribution_recommendation",
        "description": "Return citywide timing evidence, a cautious redistribution-first recommendation and the proposed controlled pilot when the user asks whether to buy or redistribute bikes.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "explain_data_limitations",
        "description": "Explain whether a requested operational fact is available. Use for station-level inventory, stock-outs, failed rentals, trip duration, station flows, routes, live demand or unsupported questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Brief description of what the user wants to know."
                }
            },
            "required": ["topic"],
        },
    },
    {
        "name": "predict_hourly_demand",
        "description": "Use the existing saved model to estimate completed rentals for a specified date, hour and weather scenario. Only call when ALL seven inputs are explicitly available from the user. Dates must be present in the original 2011–2012 dataset; this is not a live future forecast.",
        "parameters": {
            "type": "object",
            "properties": {
                "selected_date": {"type": "string", "description": "Date as YYYY-MM-DD, within 2011-01-01 to 2012-12-31."},
                "hour": {"type": "integer", "description": "Hour from 0 to 23."},
                "weather_code": {"type": "integer", "description": "Weather code from 1 to 4."},
                "temp": {"type": "number", "description": "Normalized temperature from 0 to 1."},
                "atemp": {"type": "number", "description": "Normalized feels-like temperature from 0 to 1."},
                "humidity": {"type": "number", "description": "Normalized humidity from 0 to 1."},
                "windspeed": {"type": "number", "description": "Normalized windspeed from 0 to 1."},
            },
            "required": ["selected_date", "hour", "weather_code", "temp", "atemp", "humidity", "windspeed"],
        },
    },
]

TOOL_NAMES = tuple(schema["name"] for schema in FUNCTION_SCHEMAS)
EXPECTED_ARGUMENTS = {
    "get_hourly_demand_profile": {"day_type"},
    "compare_day_types": set(),
    "get_customer_segment_summary": set(),
    "get_weather_demand_summary": set(),
    "calculate_fleet_scenario": {
        "demand_basis", "custom_hourly_rentals", "trip_minutes",
        "spatial_buffer", "maintenance_share"
    },
    "get_redistribution_recommendation": set(),
    "explain_data_limitations": {"topic"},
    "predict_hourly_demand": {
        "selected_date", "hour", "weather_code", "temp", "atemp",
        "humidity", "windspeed"
    },
}


def _require_columns(data: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(data.columns)
    if missing:
        raise ValueError(f"Project dataset is missing columns: {sorted(missing)}")


def _as_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _hour_records(profile: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
    ranked = profile.dropna().sort_values(ascending=False).head(limit)
    return [
        {
            "hour": int(hour),
            "time": f"{int(hour):02d}:00",
            "average_completed_rentals_per_hour": round(float(value), 2),
        }
        for hour, value in ranked.items()
    ]


def _tool_hourly_profile(data: pd.DataFrame, day_type: str) -> dict[str, Any]:
    _require_columns(data, {"hr", "cnt", "workingday"})
    choices = {
        "all_days": ("All days", data),
        "working_days": ("Working days", data.loc[data["workingday"] == 1]),
        "non_working_days": ("Non-working days", data.loc[data["workingday"] == 0]),
    }
    key = str(day_type).strip().lower()
    if key not in choices:
        raise ValueError("day_type must be all_days, working_days or non_working_days.")
    label, subset = choices[key]
    profile = subset.groupby("hr")["cnt"].mean().reindex(range(24))
    ranked = _hour_records(profile, limit=5)
    if not ranked:
        raise ValueError("No hourly demand observations are available for this selection.")
    return {
        "evidence_type": "observed_completed_rentals",
        "day_type": label,
        "number_of_hourly_records": int(len(subset)),
        "highest_demand_hour": ranked[0]["time"],
        "highest_average_completed_rentals_per_hour": ranked[0]["average_completed_rentals_per_hour"],
        "top_hours": ranked,
        "interpretation_limit": "These are citywide completed rentals, not station-level unmet demand.",
    }


def _complete_day_commute_shares(data: pd.DataFrame) -> dict[str, Any]:
    required = {"dteday", "hr", "workingday", "cnt"}
    if not required.issubset(data.columns):
        return {"note": "Daily commute-window shares unavailable because required date columns are missing."}
    work = data.copy()
    work["dteday"] = pd.to_datetime(work["dteday"], errors="coerce")
    work["hr"] = pd.to_numeric(work["hr"], errors="coerce")
    records: list[dict[str, Any]] = []
    for day, group in work.dropna(subset=["dteday", "hr"]).groupby("dteday"):
        if group["hr"].nunique() != 24 or group["hr"].duplicated().any():
            continue
        total = float(group["cnt"].sum())
        if total <= 0:
            continue
        commute = float(group.loc[group["hr"].isin([7, 8, 9, 17, 18, 19]), "cnt"].sum())
        records.append({
            "day_type": "working_day" if int(group["workingday"].iloc[0]) == 1 else "non_working_day",
            "share": commute / total,
        })
    daily = pd.DataFrame(records)
    if daily.empty:
        return {"note": "No complete 24-hour days were available."}
    result = {}
    for key, label in [("working_day", "working_days"), ("non_working_day", "non_working_days")]:
        vals = daily.loc[daily["day_type"] == key, "share"]
        result[label] = {
            "complete_days": int(vals.size),
            "mean_commute_window_share_pct": round(float(vals.mean() * 100), 2) if not vals.empty else None,
            "median_commute_window_share_pct": round(float(vals.median() * 100), 2) if not vals.empty else None,
        }
    result["commute_window_definition"] = "07:00–09:00 and 17:00–19:00"
    return result


def _tool_compare_day_types(data: pd.DataFrame) -> dict[str, Any]:
    _require_columns(data, {"workingday", "cnt"})
    means = data.groupby("workingday")["cnt"].mean()
    if 0 not in means.index or 1 not in means.index:
        raise ValueError("Both working-day and non-working-day rows are required.")
    non_working = float(means.loc[0])
    working = float(means.loc[1])
    difference = ((working / non_working) - 1) * 100 if non_working else None
    return {
        "evidence_type": "observed_completed_rentals",
        "working_day_average_rentals_per_hour": round(working, 2),
        "non_working_day_average_rentals_per_hour": round(non_working, 2),
        "working_day_difference_vs_non_working_pct": round(difference, 2) if difference is not None else None,
        "commute_window_daily_shares": _complete_day_commute_shares(data),
        "interpretation_limit": "Association in historical citywide completed rentals; it does not identify station movements or shortages.",
    }


def _tool_customer_segments(data: pd.DataFrame) -> dict[str, Any]:
    _require_columns(data, {"workingday", "casual", "registered"})
    grouped = data.groupby("workingday")[["casual", "registered"]].sum()
    rows = []
    for key, label in [(1, "Working day"), (0, "Non-working day")]:
        if key not in grouped.index:
            continue
        casual = float(grouped.loc[key, "casual"])
        registered = float(grouped.loc[key, "registered"])
        total = casual + registered
        rows.append({
            "day_type": label,
            "casual_rentals": int(casual),
            "registered_rentals": int(registered),
            "total_rentals": int(total),
            "casual_share_pct": round(casual / total * 100, 2) if total else None,
            "registered_share_pct": round(registered / total * 100, 2) if total else None,
        })
    return {
        "evidence_type": "observed_completed_rentals",
        "customer_segment_summary": rows,
        "interpretation_limit": "Customer shares describe completed rentals, not customer demographics or failed attempts.",
    }


def _tool_weather_summary(data: pd.DataFrame) -> dict[str, Any]:
    _require_columns(data, {"weathersit", "cnt"})
    labels = {
        1: "Clear / few clouds",
        2: "Mist / cloudy",
        3: "Light rain / snow",
        4: "Heavy rain / snow",
    }
    grouped = data.groupby("weathersit")["cnt"].agg(["mean", "median", "count"])
    clear_mean = float(grouped.loc[1, "mean"]) if 1 in grouped.index else None
    rows = []
    for code, row in grouped.iterrows():
        mean = float(row["mean"])
        rows.append({
            "weather_code": int(code),
            "weather": labels.get(int(code), f"Weather category {code}"),
            "average_completed_rentals_per_hour": round(mean, 2),
            "median_completed_rentals_per_hour": round(float(row["median"]), 2),
            "hourly_observations": int(row["count"]),
            "change_vs_clear_pct": round((mean / clear_mean - 1) * 100, 2) if clear_mean else None,
        })
    return {
        "evidence_type": "observed_association",
        "weather_summary": rows,
        "caveat": "Weather category 4 contains very few observations (three in this dataset); its average is not a reliable general estimate. These are associations, not causal effects.",
    }


def _tool_fleet_scenario(
    data: pd.DataFrame,
    demand_basis: str,
    custom_hourly_rentals: Any,
    trip_minutes: Any,
    spatial_buffer: Any,
    maintenance_share: Any,
) -> dict[str, Any]:
    demand_basis = str(demand_basis).strip().lower()
    percentiles = observed_demand_percentiles(data, year=2012)
    basis_map = {
        "observed_p90_2012": ("2012 observed P90", percentiles["p90"], 90),
        "observed_p95_2012": ("2012 observed P95", percentiles["p95"], 95),
        "observed_p99_2012": ("2012 observed P99", percentiles["p99"], 99),
        "observed_max_2012": ("2012 observed maximum", percentiles["maximum"], 100),
    }
    if demand_basis == "custom":
        demand = _as_finite_number(custom_hourly_rentals, "custom_hourly_rentals")
        if demand <= 0:
            raise ValueError("For a custom scenario, custom_hourly_rentals must be greater than zero.")
        basis_label = "Custom hourly completed-rental rate"
        percentile = None
    elif demand_basis in basis_map:
        basis_label, demand, percentile = basis_map[demand_basis]
    else:
        raise ValueError("Unsupported demand_basis.")

    trip = _as_finite_number(trip_minutes, "trip_minutes")
    buffer = _as_finite_number(spatial_buffer, "spatial_buffer")
    reserve = _as_finite_number(maintenance_share, "maintenance_share")
    scenario = estimate_fleet(
        hourly_rentals=demand,
        trip_minutes=trip,
        spatial_buffer=buffer,
        maintenance_share=reserve,
    )
    return {
        "evidence_type": "assumption_based_scenario",
        "demand_basis": basis_label,
        "observed_2012_percentile": percentile,
        "hourly_completed_rentals_used": round(float(demand), 2),
        "assumptions": {
            "trip_minutes": trip,
            "spatial_buffer": buffer,
            "maintenance_share_pct": round(reserve * 100, 2),
        },
        "bikes_in_motion": round(float(scenario["bikes_in_motion"]), 2),
        "positioned_bikes_before_maintenance_reserve": round(float(scenario["bikes_positioned"]), 2),
        "estimated_scenario_fleet": int(scenario["estimated_fleet"]),
        "formula": scenario["formula"],
        "warning": "This is not a measured fleet requirement. Trip duration, positioning buffer and maintenance share are assumptions. The dataset contains citywide completed rentals, not station-level inventory or unmet demand.",
    }


def _tool_redistribution_recommendation(data: pd.DataFrame) -> dict[str, Any]:
    timing = demand_timing_recommendations(data)
    return {
        "evidence_type": "observed_timing_plus_proposed_experiment",
        "working_day_high_demand_hours": timing["working_day_top_hours"],
        "non_working_day_high_demand_hours": timing["non_working_day_top_hours"],
        "working_day_trial_windows": timing["working_day_trial_windows"],
        "non_working_day_guidance": timing["non_working_day_trial_guidance"],
        "recommendation": "Do not recommend broad fleet expansion solely from citywide completed-rental counts. First collect station-level inventory, failed-rental and bike-health data; then compare forecast-guided redistribution with business-as-usual operations in a controlled pilot.",
        "pilot": {
            "duration": "4–6 weeks",
            "design": "Matched station clusters randomly assigned to treatment (forecast-guided redistribution) or control (existing policy), if enough independent clusters are available.",
            "primary_metric": "Empty-station minutes per station-hour during pre-defined peak windows.",
            "secondary_metrics": [
                "Failed rental attempts per 1,000 attempted rentals",
                "Completed rentals per station-hour",
                "Idle-bike hours",
                "Cost per additional successful rental",
            ],
            "guardrails": ["Driver hours and kilometres", "Safety incidents", "Bike damage", "Control-zone service quality"],
        },
        "limitation": timing["limitation"],
    }


def _tool_data_limitations(topic: str) -> dict[str, Any]:
    cleaned = str(topic).strip()[:300] or "the requested operational question"
    return {
        "requested_topic": cleaned,
        "answer": "The supplied UCI files contain citywide completed rentals. They cannot establish station-level inventory, empty-station duration, failed rental attempts, trip duration, station-to-station flows, route cost or the actual fleet required.",
        "data_needed": [
            "Time-stamped station rentable-bike and empty-dock snapshots",
            "Trip start/end station and start/end timestamps",
            "Failed or abandoned rental attempts with reasons",
            "Bike health and maintenance status",
            "Driver/vehicle rebalancing logs, distance and cost",
        ],
        "safe_next_step": "Collect these operational records and evaluate a controlled redistribution pilot. Do not infer station shortages from citywide completed-rental counts.",
    }


def _tool_prediction(
    predictor: Callable[..., dict[str, Any]] | None,
    selected_date: str,
    hour: Any,
    weather_code: Any,
    temp: Any,
    atemp: Any,
    humidity: Any,
    windspeed: Any,
) -> dict[str, Any]:
    if predictor is None:
        raise ValueError("The saved demand model is not available for prediction.")
    parsed_date = date.fromisoformat(str(selected_date))
    if not date(2011, 1, 1) <= parsed_date <= date(2012, 12, 31):
        raise ValueError("The selected date must be between 2011-01-01 and 2012-12-31.")
    hour_num = int(hour)
    weather_num = int(weather_code)
    if not 0 <= hour_num <= 23:
        raise ValueError("hour must be between 0 and 23.")
    if weather_num not in {1, 2, 3, 4}:
        raise ValueError("weather_code must be between 1 and 4.")
    values = {
        "temp": _as_finite_number(temp, "temp"),
        "atemp": _as_finite_number(atemp, "atemp"),
        "humidity": _as_finite_number(humidity, "humidity"),
        "windspeed": _as_finite_number(windspeed, "windspeed"),
    }
    if any(not 0 <= value <= 1 for value in values.values()):
        raise ValueError("Normalized weather inputs must be between 0 and 1.")
    prediction_output = predictor(
        selected_date=parsed_date,
        selected_hour=hour_num,
        weather=weather_num,
        temp=values["temp"],
        atemp=values["atemp"],
        humidity=values["humidity"],
        windspeed=values["windspeed"],
    )
    return {
        "evidence_type": "model_prediction",
        "date": parsed_date.isoformat(),
        "hour": hour_num,
        "weather_code": weather_num,
        "normalized_weather_inputs": values,
        "predicted_completed_rentals": round(float(prediction_output["prediction"]), 2),
        "model_name": str(prediction_output["model_name"]),
        "warning": "This is a what-if estimate for a date in the historical dataset range, not a live forecast or a measure of unmet demand.",
    }


def _dispatch(
    function_name: str,
    args: dict[str, Any],
    data: pd.DataFrame,
    predictor: Callable[..., dict[str, Any]] | None,
) -> dict[str, Any]:
    if function_name not in EXPECTED_ARGUMENTS:
        raise ValueError("The requested function is not allow-listed.")
    if not isinstance(args, dict):
        raise ValueError("Function arguments must be an object.")

    allowed = EXPECTED_ARGUMENTS[function_name]
    unknown = set(args) - allowed
    missing = allowed - set(args)
    if unknown:
        raise ValueError(f"Unexpected function arguments: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing function arguments: {sorted(missing)}")

    handlers = {
        "get_hourly_demand_profile": lambda: _tool_hourly_profile(data, **args),
        "compare_day_types": lambda: _tool_compare_day_types(data),
        "get_customer_segment_summary": lambda: _tool_customer_segments(data),
        "get_weather_demand_summary": lambda: _tool_weather_summary(data),
        "calculate_fleet_scenario": lambda: _tool_fleet_scenario(data, **args),
        "get_redistribution_recommendation": lambda: _tool_redistribution_recommendation(data),
        "explain_data_limitations": lambda: _tool_data_limitations(**args),
        "predict_hourly_demand": lambda: _tool_prediction(predictor, **args),
    }
    result = handlers[function_name]()
    # Ensure that only JSON-safe primitive output is sent to the model.
    json.dumps(result, allow_nan=False)
    return result


def _fallback_route(question: str) -> tuple[str, dict[str, Any]]:
    q = question.lower()
    # Station-specific / unavailable operational data takes priority.
    if any(term in q for term in [
        "station", "route", "failed rental", "stock-out", "stockout",
        "empty station", "bike availability", "trip duration", "trip length",
        "origin", "destination", "unmet demand", "real-time", "live data",
    ]):
        return "explain_data_limitations", {"topic": question}
    if any(term in q for term in ["predict", "forecast for", "estimate tomorrow"]):
        return "explain_data_limitations", {"topic": "Demand prediction requires a date, hour, weather code, temperature, feels-like temperature, humidity and windspeed. Use the separate Demand Prediction form."}
    if any(term in q for term in ["weather", "rain", "humidity", "temperature", "wind"]):
        return "get_weather_demand_summary", {}
    if any(term in q for term in ["casual", "registered", "customer", "segment", "user type"]):
        return "get_customer_segment_summary", {}
    if any(term in q for term in ["highest", "busiest", "peak", "quietest", "hourly", "time of day"]):
        if any(term in q for term in ["weekend", "non-working"]):
            return "get_hourly_demand_profile", {"day_type": "non_working_days"}
        if any(term in q for term in ["working day", "working days", "weekday"]):
            return "get_hourly_demand_profile", {"day_type": "working_days"}
        return "get_hourly_demand_profile", {"day_type": "all_days"}
    if any(term in q for term in ["working day", "working days", "weekday", "weekend", "non-working", "compare"]):
        return "compare_day_types", {}
    if any(term in q for term in ["how many bikes", "fleet size", "fleet requirement", "fleet scenario", "p95", "p90", "p99"]):
        basis = "observed_p95_2012"
        if "p90" in q:
            basis = "observed_p90_2012"
        elif "p99" in q:
            basis = "observed_p99_2012"
        elif "maximum" in q or "max" in q:
            basis = "observed_max_2012"
        trip = 20.0
        buffer = 2.0
        reserve = 0.10
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes|min)\b", q)
        if match:
            trip = float(match.group(1))
        return "calculate_fleet_scenario", {
            "demand_basis": basis,
            "custom_hourly_rentals": 0.0,
            "trip_minutes": trip,
            "spatial_buffer": buffer,
            "maintenance_share": reserve,
        }
    if any(term in q for term in ["buy", "purchase", "redistribute", "redistribution", "allocate", "recommend", "should we"]):
        return "get_redistribution_recommendation", {}
    return "explain_data_limitations", {"topic": question}


def _fallback_answer(function_name: str, result: dict[str, Any]) -> str:
    if "error" in result:
        return f"I could not complete that analysis: {result['error']}"
    if function_name == "get_hourly_demand_profile":
        return (
            f"For {result['day_type'].lower()}, the highest average completed-rental "
            f"hour is {result['highest_demand_hour']} "
            f"({result['highest_average_completed_rentals_per_hour']} rentals per hour). "
            "This is citywide historical activity, not unmet demand."
        )
    if function_name == "compare_day_types":
        return (
            f"Working days average {result['working_day_average_rentals_per_hour']} completed "
            f"rentals per hour versus {result['non_working_day_average_rentals_per_hour']} "
            f"on non-working days. The working-day difference is "
            f"{result['working_day_difference_vs_non_working_pct']}%. "
            "These are historical citywide averages."
        )
    if function_name == "get_customer_segment_summary":
        return "The customer-segment summary below compares casual and registered completed rentals by day type."
    if function_name == "get_weather_demand_summary":
        return "The weather summary below shows historical completed-rental averages by weather category. These are associations, not causal effects."
    if function_name == "calculate_fleet_scenario":
        return (
            f"Under the stated assumptions, the scenario estimate is "
            f"{result['estimated_scenario_fleet']} bikes. This is not a verified fleet requirement."
        )
    if function_name == "get_redistribution_recommendation":
        return result["recommendation"]
    if function_name == "predict_hourly_demand":
        return (
            f"The model estimates {result['predicted_completed_rentals']} completed rentals "
            f"for {result['date']} at {result['hour']:02d}:00. This is a historical-date "
            "what-if estimate, not a live forecast."
        )
    return result.get("answer", "The project data cannot answer this question directly.")


def _numeric_tokens(value: Any) -> list[float]:
    """Extract numeric values from nested tool results and answer text."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return [float(value)]
        return []
    if isinstance(value, dict):
        out: list[float] = []
        for key, item in value.items():
            out.extend(_numeric_tokens(key))
            out.extend(_numeric_tokens(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_numeric_tokens(item))
        return out
    if isinstance(value, str):
        found = re.findall(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", value)
        numbers = []
        for token in found:
            try:
                numbers.append(float(token.replace(",", "")))
            except ValueError:
                continue
        return numbers
    return []


def _answer_is_grounded(answer: str, results: list[dict[str, Any]]) -> bool:
    source_numbers: list[float] = []
    for result in results:
        source_numbers.extend(_numeric_tokens(result))
    answer_numbers = _numeric_tokens(answer)
    if not answer_numbers:
        return True
    for number in answer_numbers:
        if not any(math.isclose(number, source, rel_tol=0.0, abs_tol=0.011)
                   or math.isclose(number, source, rel_tol=0.0, abs_tol=0.51)
                   for source in source_numbers):
            return False
    return True


def _make_gemini_tool():
    from google.genai import types

    declarations = [
        types.FunctionDeclaration(
            name=schema["name"],
            description=schema["description"],
            parameters_json_schema=schema["parameters"],
        )
        for schema in FUNCTION_SCHEMAS
    ]
    return types.Tool(function_declarations=declarations)


def _ask_gemini(
    question: str,
    data: pd.DataFrame,
    api_key: str,
    model_name: str,
    predictor: Callable[..., dict[str, Any]] | None,
) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    try:
        tool = _make_gemini_tool()
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)],
        )
        response = client.models.generate_content(
            model=model_name,
            contents=[user_content],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[tool],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=list(TOOL_NAMES),
                    )
                ),
                max_output_tokens=700,
            ),
        )

        calls = list(response.function_calls or [])
        if not calls:
            raise RuntimeError("Gemini did not select an approved analysis function.")
        if len(calls) > 3:
            raise RuntimeError("Gemini returned too many function calls for one question.")

        if not response.candidates:
            raise RuntimeError("Gemini returned no usable candidate.")
        model_function_content = response.candidates[0].content

        results: list[dict[str, Any]] = []
        function_parts = []
        function_names = []
        function_arguments = []

        for call in calls:
            function_name = str(call.name or "")
            if function_name not in EXPECTED_ARGUMENTS:
                raise ValueError("Gemini selected a function that is not allow-listed.")
            arguments = dict(call.args or {})
            try:
                result = _dispatch(function_name, arguments, data, predictor)
            except Exception as exc:
                result = {"error": str(exc)}
            results.append(result)
            function_names.append(function_name)
            function_arguments.append(arguments)

            response_kwargs = {
                "name": function_name,
                "response": {"result": result},
            }
            call_id = getattr(call, "id", None)
            if call_id:
                response_kwargs["id"] = call_id
            try:
                function_parts.append(types.Part.from_function_response(**response_kwargs))
            except TypeError:
                # Compatibility with SDK releases whose helper does not expose
                # the optional function-response call ID argument.
                response_kwargs.pop("id", None)
                function_parts.append(types.Part.from_function_response(**response_kwargs))

        tool_content = types.Content(role="user", parts=function_parts)
        final_response = client.models.generate_content(
            model=model_name,
            contents=[user_content, model_function_content, tool_content],
            config=types.GenerateContentConfig(
                system_instruction=FINAL_ANSWER_INSTRUCTION,
                max_output_tokens=650,
            ),
        )
        answer = (final_response.text or "").strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty final answer.")

        grounded = _answer_is_grounded(answer, results)
        if not grounded:
            answer = (
                "I retrieved the relevant project evidence, but the generated explanation "
                "included a numeric claim that I could not match to the tool output. "
                "Please use the verified function result shown below."
            )

        return {
            "mode": "Gemini NLP + manual function calling",
            "function_name": function_names[0] if len(function_names) == 1 else function_names,
            "arguments": function_arguments[0] if len(function_arguments) == 1 else function_arguments,
            "result": results[0] if len(results) == 1 else results,
            "answer": answer,
            "grounded": grounded,
            "fallback_reason": None,
        }
    finally:
        client.close()


def answer_business_question(
    question: str,
    data: pd.DataFrame,
    api_key: str | None = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    predictor: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Answer a business question using Gemini tool selection when a key is present.
    Falls back to deterministic routing if the key is missing or the API fails.
    """
    cleaned = str(question or "").strip()
    if not cleaned:
        raise ValueError("Please enter a business question.")
    if len(cleaned) > MAX_QUESTION_LENGTH:
        raise ValueError(f"Please keep the question under {MAX_QUESTION_LENGTH} characters.")

    key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if key:
        try:
            return _ask_gemini(cleaned, data, key, model_name, predictor)
        except Exception as exc:
            fallback_reason = f"Gemini request failed; deterministic fallback was used ({type(exc).__name__}: {str(exc)[:240]})."
    else:
        fallback_reason = "Gemini API key is not configured; deterministic fallback was used."

    function_name, arguments = _fallback_route(cleaned)
    try:
        result = _dispatch(function_name, arguments, data, predictor)
    except Exception as exc:
        function_name = "explain_data_limitations"
        arguments = {"topic": cleaned}
        result = _dispatch(function_name, arguments, data, predictor)
        fallback_reason += f" The initial local route could not run ({type(exc).__name__})."

    return {
        "mode": "Rule-based fallback",
        "function_name": function_name,
        "arguments": arguments,
        "result": result,
        "answer": _fallback_answer(function_name, result),
        "grounded": True,
        "fallback_reason": fallback_reason,
    }
