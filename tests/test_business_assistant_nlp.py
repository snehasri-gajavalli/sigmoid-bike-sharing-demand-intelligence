
import pandas as pd
import pytest

from business_assistant_nlp import (
    answer_business_question,
    _dispatch,
    _tool_hourly_profile,
    _tool_compare_day_types,
    _tool_customer_segments,
    _tool_weather_summary,
    _tool_fleet_scenario,
    _tool_prediction,
)


@pytest.fixture
def sample_hour():
    rows = []

    # One complete working day and one non-working day.
    for day, workingday in [
        ("2012-01-02", 1),
        ("2012-01-07", 0)
    ]:
        for hour in range(24):
            cnt = (
                500 if (workingday and hour == 17)
                else 300 if (not workingday and hour == 13)
                else 100 + hour
            )

            rows.append({
                "dteday": day,
                "hr": hour,
                "workingday": workingday,
                "cnt": cnt,
                "casual": int(
                    cnt * (0.2 if workingday else 0.5)
                ),
                "registered": cnt - int(
                    cnt * (0.2 if workingday else 0.5)
                ),
                "weathersit": 1 if hour % 2 == 0 else 2,
                "yr": 1,
            })

    return pd.DataFrame(rows)


def test_hourly_profile_selects_working_peak(sample_hour):
    result = _tool_hourly_profile(
        sample_hour, "working_days"
    )
    assert result["highest_demand_hour"] == "17:00"
    assert result["evidence_type"] == "observed_completed_rentals"


def test_day_type_comparison_returns_two_averages(sample_hour):
    result = _tool_compare_day_types(sample_hour)

    assert result["working_day_average_rentals_per_hour"] > 0
    assert result["non_working_day_average_rentals_per_hour"] > 0
    assert "commute_window_daily_shares" in result


def test_customer_segment_shares_sum_to_100(sample_hour):
    result = _tool_customer_segments(sample_hour)

    assert len(result["customer_segment_summary"]) == 2

    for row in result["customer_segment_summary"]:
        assert (
            row["casual_share_pct"]
            + row["registered_share_pct"]
        ) == pytest.approx(100, abs=0.02)


def test_weather_summary_has_categories(sample_hour):
    result = _tool_weather_summary(sample_hour)

    assert len(result["weather_summary"]) == 2
    assert all(
        row["hourly_observations"] > 0
        for row in result["weather_summary"]
    )


def test_fleet_scenario_is_explicitly_assumption_based(sample_hour):
    result = _tool_fleet_scenario(
        sample_hour,
        "observed_p95_2012",
        0,
        20,
        2,
        0.10,
    )

    assert result["estimated_scenario_fleet"] > 0
    assert result["evidence_type"] == "assumption_based_scenario"
    assert "not a measured fleet requirement" in result["warning"]


def test_dispatch_rejects_unapproved_function(sample_hour):
    with pytest.raises(ValueError, match="not allow-listed"):
        _dispatch(
            "run_arbitrary_python",
            {},
            sample_hour,
            None
        )


def test_dispatch_rejects_unexpected_arguments(sample_hour):
    with pytest.raises(ValueError, match="Unexpected"):
        _dispatch(
            "compare_day_types",
            {"other": 1},
            sample_hour,
            None
        )


def test_prediction_tool_validates_inputs_and_calls_model():
    observed = {}

    def predictor(**kwargs):
        observed.update(kwargs)
        return {
            "prediction": 321.5,
            "model_name": "Random Forest"
        }

    result = _tool_prediction(
        predictor,
        "2012-06-15",
        8,
        1,
        0.5,
        0.5,
        0.5,
        0.2,
    )

    assert result["predicted_completed_rentals"] == 321.5
    assert observed["selected_hour"] == 8
    assert observed["weather"] == 1


def test_fallback_routes_peak_question_and_marks_fallback(
    sample_hour, monkeypatch
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    result = answer_business_question(
        "When is demand highest on working days?",
        sample_hour,
        api_key=None,
    )

    assert result["mode"] == "Rule-based fallback"
    assert result["function_name"] == "get_hourly_demand_profile"
    assert result["result"]["highest_demand_hour"] == "17:00"


def test_fallback_refuses_station_shortage_inference(
    sample_hour, monkeypatch
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    result = answer_business_question(
        "Which station is running out of bikes?",
        sample_hour,
        api_key=None,
    )

    assert result["function_name"] == "explain_data_limitations"
    assert "station-level" in result["result"]["answer"]
