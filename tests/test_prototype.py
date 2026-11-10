
        # ============================================================
# AUTOMATED TESTS - SIGMOID BIKE SHARING CASE STUDY
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fleet_planning import (
    estimate_fleet,
    observed_demand_percentiles,
    fleet_sensitivity,
    demand_timing_recommendations,
    plan_station_transfers,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "hour.csv"

MODEL_BUNDLE_PATH = (
    BASE_DIR / "models" / "bike_demand_model_bundle.pkl"
)

LEGACY_MODEL_PATH = (
    BASE_DIR / "models" / "bike_demand_random_forest.pkl"
)

FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]


# ============================================================
# 2. FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def dataset():
    assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def model_bundle():
    """
    Bundle-specific tests are skipped when only the legacy model exists.
    This avoids treating the absence of the newer bundle as a test failure.
    """
    if not MODEL_BUNDLE_PATH.exists():
        pytest.skip(
            "New model bundle is not present. "
            "Bundle-specific tests are skipped; legacy model is checked separately."
        )

    bundle = joblib.load(MODEL_BUNDLE_PATH)

    assert isinstance(bundle, dict), "Model bundle must be a dictionary."
    return bundle


@pytest.fixture
def sample_station_data():
    return pd.DataFrame({
        "station_id": ["STN_001", "STN_002", "STN_003"],
        "available_bikes": [30, 3, 15],
        "forecast_rentals_per_hour": [5, 12, 8],
    })


# ============================================================
# 3. DATASET VALIDATION
# ============================================================

def test_dataset_exists():
    assert DATA_PATH.exists()


def test_dataset_not_empty(dataset):
    assert not dataset.empty


def test_dataset_has_required_features(dataset):
    missing = [column for column in FEATURES if column not in dataset.columns]
    assert not missing, f"Missing feature columns: {missing}"


def test_target_column_exists(dataset):
    assert "cnt" in dataset.columns


def test_target_values_are_valid(dataset):
    assert dataset["cnt"].notna().all()
    assert (dataset["cnt"] >= 0).all()


def test_total_rentals_equal_customer_segments(dataset):
    assert "casual" in dataset.columns
    assert "registered" in dataset.columns
    assert np.allclose(
        dataset["cnt"],
        dataset["casual"] + dataset["registered"],
    )


def test_hour_values_are_valid(dataset):
    assert dataset["hr"].between(0, 23).all()


def test_temperature_values_are_valid(dataset):
    assert dataset["temp"].between(0, 1).all()
    assert dataset["atemp"].between(0, 1).all()


def test_expected_dataset_size_and_total(dataset):
    # These assertions intentionally lock the supplied UCI hour.csv version.
    assert len(dataset) == 17379
    assert dataset["cnt"].sum() == 3292679


# ============================================================
# 4. MODEL ARTIFACT VALIDATION
# ============================================================

def test_at_least_one_model_artifact_exists():
    """
    The project may contain either the newer bundle or the legacy pickle.
    The app prefers the bundle and supports the legacy file only when its
    fitted feature names can be recovered.
    """
    assert MODEL_BUNDLE_PATH.exists() or LEGACY_MODEL_PATH.exists(), (
        "No model artifact found. Expected either "
        "models/bike_demand_model_bundle.pkl or "
        "models/bike_demand_random_forest.pkl."
    )


def test_model_bundle_loads(model_bundle):
    assert "model_name" in model_bundle
    assert "feature_columns" in model_bundle

    if model_bundle["model_name"] == "Random Forest":
        assert model_bundle.get("model") is not None
        assert len(model_bundle["feature_columns"]) > 0

    elif model_bundle["model_name"] == "Hourly + working-day mean baseline":
        assert model_bundle.get("baseline_table") is not None

    else:
        pytest.fail(
            f"Unsupported model name: {model_bundle.get('model_name')}"
        )


def test_model_feature_columns_are_valid(model_bundle):
    feature_columns = model_bundle["feature_columns"]

    assert isinstance(feature_columns, (list, tuple, np.ndarray))
    assert len(feature_columns) > 0
    assert len(feature_columns) == len(set(feature_columns))


def test_saved_random_forest_can_predict(model_bundle):
    if model_bundle["model_name"] != "Random Forest":
        pytest.skip("Selected model is not Random Forest.")

    model = model_bundle["model"]
    feature_columns = list(model_bundle["feature_columns"])

    # A zero-valued encoded row checks the model interface and output type.
    sample = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns,
    )

    prediction = np.asarray(model.predict(sample))

    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_legacy_model_loads_if_present():
    if not LEGACY_MODEL_PATH.exists():
        pytest.skip("Legacy Random Forest pickle is not present.")

    model = joblib.load(LEGACY_MODEL_PATH)
    assert model is not None
    assert hasattr(model, "predict")


# ============================================================
# 5. FLEET CALCULATION TESTS
# ============================================================

def test_fleet_calculation():
    result = estimate_fleet(
        hourly_rentals=702.25,
        trip_minutes=45,
        spatial_buffer=1.2,
        maintenance_share=0.10,
    )

    expected_in_motion = 702.25 * 45 / 60
    expected_positioned = expected_in_motion * 1.2
    expected_fleet = int(np.ceil(expected_positioned / 0.90))

    assert result["bikes_in_motion"] == pytest.approx(expected_in_motion)
    assert result["bikes_positioned"] == pytest.approx(expected_positioned)
    assert result["estimated_fleet"] == expected_fleet


def test_zero_demand():
    result = estimate_fleet(
        hourly_rentals=0,
        trip_minutes=20,
        spatial_buffer=2,
        maintenance_share=0.10,
    )

    assert result["estimated_fleet"] == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "hourly_rentals": -10,
            "trip_minutes": 20,
            "spatial_buffer": 2,
            "maintenance_share": 0.10,
        },
        {
            "hourly_rentals": 500,
            "trip_minutes": 0,
            "spatial_buffer": 2,
            "maintenance_share": 0.10,
        },
        {
            "hourly_rentals": 500,
            "trip_minutes": 20,
            "spatial_buffer": 0,
            "maintenance_share": 0.10,
        },
        {
            "hourly_rentals": 500,
            "trip_minutes": 20,
            "spatial_buffer": 2,
            "maintenance_share": 1.0,
        },
        {
            "hourly_rentals": 500,
            "trip_minutes": 20,
            "spatial_buffer": 2,
            "maintenance_share": -0.1,
        },
    ],
)
def test_invalid_fleet_inputs(arguments):
    with pytest.raises(ValueError):
        estimate_fleet(**arguments)


# ============================================================
# 6. OBSERVED DEMAND PERCENTILES
# ============================================================

def test_observed_demand_percentiles(dataset):
    result = observed_demand_percentiles(dataset, year=2012)

    for key in ("p90", "p95", "p99", "maximum", "n_hours"):
        assert key in result

    assert result["p90"] <= result["p95"]
    assert result["p95"] <= result["p99"]
    assert result["p99"] <= result["maximum"]
    assert result["n_hours"] > 0


# ============================================================
# 7. FLEET SENSITIVITY
# ============================================================

def test_fleet_sensitivity():
    # The first argument is positional: fleet_sensitivity does not accept
    # an hourly_demand keyword in the project's current implementation.
    result = fleet_sensitivity(
        500,
        trip_minutes_values=(10, 20),
        spatial_buffer_values=(1.5, 2.0),
        maintenance_values=(0.05, 0.10),
    )

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert len(result) == 8


# ============================================================
# 8. DEMAND TIMING RECOMMENDATIONS
# ============================================================

def test_demand_timing_recommendations(dataset):
    result = demand_timing_recommendations(dataset)

    assert isinstance(result, dict)
    assert "working_day_top_hours" in result
    assert "non_working_day_top_hours" in result
    assert len(result["working_day_top_hours"]) > 0
    assert len(result["non_working_day_top_hours"]) > 0


# ============================================================
# 9. STATION TRANSFER PLANNING
# ============================================================

def test_station_transfer_planning(sample_station_data):
    result = plan_station_transfers(
        sample_station_data,
        forecast_horizon_hours=2,
        reserve_multiplier=1.15,
    )

    assert isinstance(result, pd.DataFrame)


def test_station_transfer_invalid_columns():
    invalid_data = pd.DataFrame({
        "station": ["A", "B"],
        "bikes": [10, 20],
    })

    with pytest.raises((ValueError, KeyError)):
        plan_station_transfers(
            invalid_data,
            forecast_horizon_hours=2,
            reserve_multiplier=1.15,
        )


# ============================================================
# 10. CUSTOMER SEGMENT VALIDATION
# ============================================================

def test_customer_segments(dataset):
    assert "casual" in dataset.columns
    assert "registered" in dataset.columns
    assert (dataset["casual"] >= 0).all()
    assert (dataset["registered"] >= 0).all()


# ============================================================
# END OF TESTS
# ============================================================
