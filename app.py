from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from fleet_planning import (
    estimate_fleet,
    observed_demand_percentiles,
    fleet_sensitivity,
    demand_timing_recommendations,
    plan_station_transfers,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bike Sharing Demand Intelligence",
    page_icon="🚲",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hour.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_BUNDLE_PATH = MODEL_DIR / "bike_demand_model_bundle.pkl"
LEGACY_MODEL_PATH = MODEL_DIR / "bike_demand_random_forest.pkl"

RAW_FEATURE_COLUMNS = [
    "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum",
    "windspeed", "year", "month", "day", "dayofweek",
]
CATEGORICAL_COLUMNS = [
    "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "year", "month", "day", "dayofweek",
]
NUMERIC_COLUMNS = ["temp", "atemp", "hum", "windspeed"]

WEATHER_LABELS = {
    1: "Clear / Few clouds",
    2: "Mist / Cloudy",
    3: "Light rain / snow",
    4: "Heavy rain / snow",
}

sns.set_theme(style="whitegrid")


# ============================================================
# 2. DATA LOADING
# ============================================================

@st.cache_data
def load_data(path_string: str) -> pd.DataFrame:
    """Load original UCI hour.csv and derive notebook calendar features."""
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(
            f"Hourly dataset not found: {path}. Expected data/hour.csv."
        )

    data = pd.read_csv(path)

    # Validate only columns that are present in the original UCI file.
    original_columns = [
        "dteday", "season", "yr", "mnth", "hr", "holiday",
        "weekday", "workingday", "weathersit", "temp", "atemp",
        "hum", "windspeed", "casual", "registered", "cnt",
    ]
    missing = [col for col in original_columns if col not in data.columns]
    if missing:
        raise ValueError(f"hour.csv is missing required columns: {missing}")

    data["dteday"] = pd.to_datetime(data["dteday"], errors="coerce")
    if data["dteday"].isna().any():
        raise ValueError("Some dteday values could not be parsed.")

    data = data.sort_values(["dteday", "hr"]).reset_index(drop=True)

    # These four calendar fields are engineered in notebook 04.
    data["year"] = data["dteday"].dt.year
    data["month"] = data["dteday"].dt.month
    data["day"] = data["dteday"].dt.day
    data["dayofweek"] = data["dteday"].dt.dayofweek

    return data


try:
    hour_df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error("The application could not load the hourly dataset.")
    st.exception(exc)
    st.stop()


# ============================================================
# 3. MODEL LOADING AND NOTEBOOK-MATCHING PREPROCESSING
# ============================================================

def make_model_matrix(raw_data: pd.DataFrame, feature_columns) -> pd.DataFrame:
    """
    Reproduce notebook 04 exactly:
    pandas get_dummies for categorical columns, concatenate numeric columns,
    then align the result to the training feature-column order.
    """
    missing = [
        col for col in RAW_FEATURE_COLUMNS
        if col not in raw_data.columns
    ]
    if missing:
        raise ValueError(f"Prediction input is missing raw features: {missing}")

    categorical = pd.get_dummies(
        raw_data[CATEGORICAL_COLUMNS].astype("category"),
        columns=CATEGORICAL_COLUMNS,
        dtype=int,
    )
    numeric = raw_data[NUMERIC_COLUMNS].reset_index(drop=True)

    matrix = pd.concat(
        [numeric, categorical.reset_index(drop=True)],
        axis=1,
    )
    return matrix.reindex(columns=list(feature_columns), fill_value=0)


@st.cache_resource
def load_model_artifact():
    """
    Load the current notebook bundle first.

    A legacy 12-feature Random Forest is supported only as a fallback.
    The notebook-generated bundle is the preferred and reproducible artifact.
    """
    if MODEL_BUNDLE_PATH.exists():
        bundle = joblib.load(MODEL_BUNDLE_PATH)

        if not isinstance(bundle, dict):
            raise ValueError("The model bundle must be a dictionary.")

        model_name = bundle.get("model_name")
        input_format = bundle.get("input_format")

        if model_name == "Random Forest":
            model = bundle.get("model")
            columns = bundle.get("feature_columns")

            if model is None or not hasattr(model, "predict"):
                raise ValueError("The bundle does not contain a fitted prediction model.")
            if not columns:
                raise ValueError("The bundle has no encoded feature-column schema.")
            if input_format != "onehot":
                raise ValueError(
                    "The Random Forest bundle must use input_format='onehot'. "
                    "Rerun the final save cell in notebook 04."
                )

        elif model_name == "Hourly + working-day mean baseline":
            if bundle.get("baseline_table") is None:
                raise ValueError("The baseline bundle is missing baseline_table.")
        else:
            raise ValueError(f"Unsupported model name in bundle: {model_name}")

        return bundle, "notebook_bundle"

    # Backward compatibility for the old 12-raw-feature pickle.
    if LEGACY_MODEL_PATH.exists():
        legacy_model = joblib.load(LEGACY_MODEL_PATH)
        names = list(getattr(legacy_model, "feature_names_in_", []))

        legacy_features = [
            "season", "yr", "mnth", "hr", "holiday", "weekday",
            "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
        ]

        if names != legacy_features:
            raise ValueError(
                "The legacy model does not have the expected 12 raw feature names. "
                "Run notebook 04 and create the new model bundle."
            )

        legacy_bundle = {
            "model_name": "Random Forest",
            "model": legacy_model,
            "feature_columns": legacy_features,
            "input_format": "raw_legacy",
        }
        return legacy_bundle, "legacy_model"

    raise FileNotFoundError(
        "No model artifact found. Run all cells in "
        "notebooks/04_demand_forecasting_COMPLETE.ipynb, including its final save cell."
    )


try:
    model_artifact, model_artifact_source = load_model_artifact()
    model_load_error = None
except Exception as exc:
    model_artifact = None
    model_artifact_source = None
    model_load_error = str(exc)


# ============================================================
# 4. SHARED HELPERS
# ============================================================

def format_hour(hour: int) -> str:
    return f"{int(hour):02d}:00"


def get_hourly_profile(data: pd.DataFrame) -> pd.Series:
    return data.groupby("hr")["cnt"].mean().reindex(range(24))


def get_weather_summary(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby("weathersit")["cnt"].agg(["mean", "median", "count"])


def build_prediction_row(
    selected_date,
    selected_hour: int,
    weather: int,
    temp: float,
    atemp: float,
    humidity: float,
    windspeed: float,
) -> pd.DataFrame:
    """
    Build one raw model row. Calendar values are looked up from the supplied
    dataset date so weekday, holiday, workingday and season match its encoding.
    """
    selected_timestamp = pd.Timestamp(selected_date)

    date_rows = hour_df.loc[
        hour_df["dteday"].dt.normalize() == selected_timestamp.normalize()
    ]

    if date_rows.empty:
        raise ValueError(
            "The selected date is not represented in the source dataset. "
            "Choose a date between 2011-01-01 and 2012-12-31."
        )

    calendar = date_rows.iloc[0]

    row = {
        "season": int(calendar["season"]),
        "yr": int(calendar["yr"]),
        "mnth": int(calendar["mnth"]),
        "hr": int(selected_hour),
        "holiday": int(calendar["holiday"]),
        "weekday": int(calendar["weekday"]),
        "workingday": int(calendar["workingday"]),
        "weathersit": int(weather),
        "temp": float(temp),
        "atemp": float(atemp),
        "hum": float(humidity),
        "windspeed": float(windspeed),
        "year": int(selected_timestamp.year),
        "month": int(selected_timestamp.month),
        "day": int(selected_timestamp.day),
        "dayofweek": int(selected_timestamp.dayofweek),
    }

    return pd.DataFrame([row], columns=RAW_FEATURE_COLUMNS)


def predict_demand(input_row: pd.DataFrame):
    if model_artifact is None:
        raise RuntimeError(model_load_error or "Model is unavailable.")

    model_name = model_artifact["model_name"]

    if model_name == "Random Forest":
        if model_artifact.get("input_format") == "onehot":
            matrix = make_model_matrix(
                input_row,
                model_artifact["feature_columns"],
            )
        elif model_artifact.get("input_format") == "raw_legacy":
            matrix = input_row.reindex(
                columns=model_artifact["feature_columns"]
            )
        else:
            raise ValueError("Unknown model input format.")

        prediction = float(model_artifact["model"].predict(matrix)[0])
        return max(0.0, prediction), model_name

    if model_name == "Hourly + working-day mean baseline":
        baseline = model_artifact["baseline_table"]
        match = baseline[
            (baseline["hr"] == int(input_row.iloc[0]["hr"]))
            & (baseline["workingday"] == int(input_row.iloc[0]["workingday"]))
        ]

        if not match.empty:
            prediction = float(match.iloc[0]["baseline_prediction"])
        else:
            prediction = float(model_artifact["fallback_mean"])

        return max(0.0, prediction), model_name

    raise ValueError(f"Unsupported selected model: {model_name}")


# ============================================================
# 5. HEADER, STYLE AND NAVIGATION
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 750;
        color: #E07A45;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #888888;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.08);
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🚲 Bike Sharing Demand Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Historical demand analytics, hourly forecasting '
    'and operational decision support</div>',
    unsafe_allow_html=True,
)

if model_artifact is None:
    st.warning(
        "Demand prediction is unavailable until a compatible model artifact is found. "
        "The analytical pages remain available."
    )
    with st.expander("Model loading details"):
        st.code(model_load_error or "Unknown model-loading error.")
elif model_artifact_source == "legacy_model":
    st.warning(
        "The dashboard is using the older 12-feature model as a fallback. "
        "Run the final save cell in the current forecasting notebook to create "
        "the preferred notebook-matched model bundle."
    )

st.sidebar.title("🚲 Navigation")
page = st.sidebar.radio(
    "Select a section",
    [
        "Demand Overview",
        "Hourly Analysis",
        "Customer Segmentation",
        "Demand Prediction",
        "Fleet Planning & Experiment",
        "Business Assistant",
    ],
)
st.sidebar.divider()
st.sidebar.caption("Python • Pandas • Scikit-learn • Streamlit")


# ============================================================
# 6. DEMAND OVERVIEW
# ============================================================

if page == "Demand Overview":
    st.header("Demand Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total completed rentals", f"{hour_df['cnt'].sum():,.0f}")
    c2.metric("Average hourly rentals", f"{hour_df['cnt'].mean():.1f}")
    c3.metric("Maximum hourly rentals", f"{hour_df['cnt'].max():,.0f}")
    c4.metric("Hourly observations", f"{len(hour_df):,}")

    st.divider()
    st.subheader("Average demand across 24 hours")

    profile = get_hourly_profile(hour_df)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(profile.index, profile.values, marker="o")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average completed rentals")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Working day versus non-working day")
    comparison = hour_df.groupby("workingday")["cnt"].mean()
    c1, c2 = st.columns(2)
    c1.metric("Non-working-day hourly average", f"{comparison.loc[0]:.2f}")
    c2.metric("Working-day hourly average", f"{comparison.loc[1]:.2f}")

    st.caption(
        "These are completed rentals. They do not measure failed rental attempts, "
        "unmet demand or simultaneous bike occupancy."
    )


# ============================================================
# 7. HOURLY ANALYSIS
# ============================================================

elif page == "Hourly Analysis":
    st.header("Hourly Demand Analysis")

    day_type = st.selectbox(
        "Select day type",
        ["All Days", "Working Days", "Non-Working Days"],
    )
    filtered = hour_df.copy()

    if day_type == "Working Days":
        filtered = filtered[filtered["workingday"] == 1]
    elif day_type == "Non-Working Days":
        filtered = filtered[filtered["workingday"] == 0]

    hourly = (
        filtered.groupby("hr")["cnt"]
        .mean()
        .reindex(range(24))
        .rename("Average Rentals")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.barplot(data=hourly, x="hr", y="Average Rentals", ax=ax)
    ax.set_title(f"Hourly demand — {day_type}")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average completed rentals")
    ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(hourly.round(2), use_container_width=True)

    peak_row = hourly.loc[hourly["Average Rentals"].idxmax()]
    st.success(
        f"Highest average demand is at {format_hour(peak_row['hr'])}: "
        f"{peak_row['Average Rentals']:.1f} completed rentals per hour."
    )

    st.subheader("Working-day versus non-working-day hourly profile")
    heatmap_data = (
        hour_df.groupby(["hr", "workingday"])["cnt"]
        .mean()
        .unstack()
        .reindex(range(24))
    )
    heatmap_data.columns = ["Non-Working Day", "Working Day"]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_xlabel("Day type")
    ax.set_ylabel("Hour")
    st.pyplot(fig)
    plt.close(fig)

    difference = heatmap_data["Working Day"] - heatmap_data["Non-Working Day"]
    difference_table = pd.DataFrame({
        "Working Day": heatmap_data["Working Day"],
        "Non-Working Day": heatmap_data["Non-Working Day"],
        "Difference": difference,
        "Percentage Difference": (
            difference / heatmap_data["Non-Working Day"].replace(0, np.nan) * 100
        ),
    })
    st.subheader("Hourly demand difference")
    st.dataframe(difference_table.round(2), use_container_width=True)


# ============================================================
# 8. CUSTOMER SEGMENTATION
# ============================================================

elif page == "Customer Segmentation":
    st.header("Customer Segmentation")

    contribution = hour_df.groupby("workingday")[["casual", "registered"]].sum()
    contribution.index = ["Non-Working Day", "Working Day"]
    contribution["total"] = contribution["casual"] + contribution["registered"]
    contribution["Casual (%)"] = contribution["casual"] / contribution["total"] * 100
    contribution["Registered (%)"] = (
        contribution["registered"] / contribution["total"] * 100
    )

    st.subheader("Customer segment contribution")
    st.dataframe(contribution.round(2), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    contribution[["Casual (%)", "Registered (%)"]].plot(
        kind="bar", stacked=True, ax=ax
    )
    ax.set_ylabel("Share of rentals (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    ax.set_title("Customer contribution by day type")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Hourly customer behaviour")
    segment = st.selectbox("Customer segment", ["Registered", "Casual"])
    column = "registered" if segment == "Registered" else "casual"

    user_hourly = (
        hour_df.groupby(["workingday", "hr"])[column]
        .mean()
        .reset_index()
    )
    user_hourly["Day Type"] = user_hourly["workingday"].map({
        0: "Non-Working Day",
        1: "Working Day",
    })

    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.lineplot(
        data=user_hourly,
        x="hr",
        y=column,
        hue="Day Type",
        marker="o",
        ax=ax,
    )
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour")
    ax.set_ylabel(f"Average {segment.lower()} rentals")
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# 9. DEMAND PREDICTION
# ============================================================

elif page == "Demand Prediction":
    st.header("Hourly Bike Demand Prediction")
    st.write(
        "Choose a date, hour and weather conditions. Calendar fields are "
        "derived from the selected date and the original dataset."
    )

    if model_artifact is None:
        st.error("Prediction cannot run until a compatible model is available.")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                selected_date = st.date_input(
                    "Date",
                    value=pd.Timestamp("2012-06-15").date(),
                    min_value=pd.Timestamp("2011-01-01").date(),
                    max_value=pd.Timestamp("2012-12-31").date(),
                )
                selected_hour = st.slider("Hour of day", 0, 23, 8)
                weather = st.selectbox(
                    "Weather situation",
                    [1, 2, 3, 4],
                    format_func=lambda x: WEATHER_LABELS[x],
                )

            with col2:
                temp = st.slider("Temperature (scaled 0–1)", 0.0, 1.0, 0.5)
                atemp = st.slider("Feels-like temperature (scaled 0–1)", 0.0, 1.0, 0.5)
                humidity = st.slider("Humidity (scaled 0–1)", 0.0, 1.0, 0.5)
                windspeed = st.slider("Windspeed (scaled 0–1)", 0.0, 1.0, 0.2)

            submitted = st.form_submit_button("Predict Bike Demand")

        if submitted:
            try:
                input_row = build_prediction_row(
                    selected_date=selected_date,
                    selected_hour=selected_hour,
                    weather=weather,
                    temp=temp,
                    atemp=atemp,
                    humidity=humidity,
                    windspeed=windspeed,
                )

                prediction, model_name = predict_demand(input_row)

                st.success(f"Prediction generated using: {model_name}")
                st.metric("Predicted completed rentals", f"{prediction:.0f}")
                st.caption(
                    "This is a model estimate of completed rentals, not a guarantee "
                    "of future demand or a measure of unmet demand. Weather and "
                    "continuous weather variables are user-supplied scenario inputs."
                )

                with st.expander("Inspect model input"):
                    st.dataframe(input_row, use_container_width=True)

            except Exception as exc:
                st.error("Prediction failed.")
                st.exception(exc)


# ============================================================
# 10. FLEET PLANNING & REDISTRIBUTION EXPERIMENT
# ============================================================

elif page == "Fleet Planning & Experiment":
    st.header("Fleet Planning & Redistribution")
    st.write(
        "Explore fleet scenarios using Little's Law, sensitivity to assumptions "
        "and station-level transfer shortlists when operational station data exists."
    )

    percentiles = observed_demand_percentiles(hour_df, year=2012)

    st.subheader("1. Historical demand reference — 2012")
    cols = st.columns(4)
    cols[0].metric("P90 hourly rentals", f"{percentiles['p90']:.0f}")
    cols[1].metric("P95 hourly rentals", f"{percentiles['p95']:.0f}")
    cols[2].metric("P99 hourly rentals", f"{percentiles['p99']:.0f}")
    cols[3].metric("Maximum hourly rentals", f"{percentiles['maximum']:.0f}")
    st.caption(
        f"Based on {percentiles['n_hours']:,} recorded hourly observations. "
        "These are completed rentals, not unserved demand."
    )

    st.subheader("2. Fleet scenario using Little's Law")
    st.latex(r"L = \lambda W")
    st.write(
        "Average bikes in motion = hourly rental-start rate × average trip duration. "
        "Spatial positioning and maintenance reserves are additional assumptions."
    )

    basis_label = st.selectbox(
        "Demand basis",
        ["2012 P90", "2012 P95", "2012 P99", "2012 maximum", "Custom hourly rentals"],
        index=1,
    )
    basis_map = {
        "2012 P90": percentiles["p90"],
        "2012 P95": percentiles["p95"],
        "2012 P99": percentiles["p99"],
        "2012 maximum": percentiles["maximum"],
    }

    if basis_label == "Custom hourly rentals":
        demand_value = st.number_input(
            "Custom completed rentals per hour",
            min_value=1.0,
            max_value=10000.0,
            value=float(round(percentiles["p95"])),
            step=10.0,
        )
    else:
        demand_value = float(basis_map[basis_label])

    c1, c2, c3 = st.columns(3)
    trip_minutes = c1.slider("Average trip duration (minutes)", 5, 120, 20, 5)
    spatial_buffer = c2.slider("Spatial positioning buffer", 1.0, 4.0, 2.0, 0.25)
    maintenance_share = c3.slider(
        "Fleet unavailable for maintenance (%)", 0, 40, 10, 5
    ) / 100.0

    fleet_result = estimate_fleet(
        hourly_rentals=demand_value,
        trip_minutes=trip_minutes,
        spatial_buffer=spatial_buffer,
        maintenance_share=maintenance_share,
    )

    cols = st.columns(3)
    cols[0].metric("Bikes in motion", f"{fleet_result['bikes_in_motion']:.1f}")
    cols[1].metric("Positioned bikes", f"{fleet_result['bikes_positioned']:.1f}")
    cols[2].metric("Scenario fleet", f"{fleet_result['estimated_fleet']:,}")

    st.code(
        f"ceil({demand_value:.1f} × {trip_minutes}/60 × "
        f"{spatial_buffer:.2f} ÷ (1 - {maintenance_share:.2f})) "
        f"= {fleet_result['estimated_fleet']:,} bikes",
        language="text",
    )
    st.warning(
        "Scenario only. Trip duration, spatial buffer and maintenance share are "
        "assumptions. Citywide hourly rental counts cannot establish actual fleet "
        "requirement or identify station shortages."
    )

    with st.expander("Sensitivity analysis"):
        sensitivity = fleet_sensitivity(
            demand_value,
            trip_minutes_values=(10, 20, 30),
            spatial_buffer_values=(1.5, 2.0, 3.0),
            maintenance_values=(0.05, 0.10, 0.15),
        )
        st.dataframe(sensitivity.round(1), use_container_width=True)
        st.download_button(
            "Download fleet sensitivity CSV",
            sensitivity.to_csv(index=False).encode("utf-8"),
            file_name="fleet_sensitivity.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("3. Demand-informed operating windows")
    timing = demand_timing_recommendations(hour_df)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Working-day top demand hours**")
        st.dataframe(
            pd.DataFrame(timing["working_day_top_hours"]),
            use_container_width=True,
        )
    with c2:
        st.markdown("**Non-working-day top demand hours**")
        st.dataframe(
            pd.DataFrame(timing["non_working_day_top_hours"]),
            use_container_width=True,
        )

    for recommendation in timing["working_day_trial_windows"]:
        st.write(f"- {recommendation}")
    st.info(timing["non_working_day_trial_guidance"])

    st.divider()
    st.subheader("4. Optional station-level transfer shortlist")
    st.write(
        "Upload a current station snapshot CSV with columns: "
        "`station_id`, `available_bikes`, `forecast_rentals_per_hour`. "
        "This creates a heuristic donor/receiver shortlist, not route optimisation."
    )

    station_file = st.file_uploader(
        "Upload station inventory and forecast CSV",
        type=["csv"],
        key="station_snapshot_upload",
    )

    if station_file is not None:
        try:
            station_snapshot = pd.read_csv(station_file)
            horizon = st.number_input(
                "Planning horizon (hours)",
                min_value=0.5,
                max_value=12.0,
                value=2.0,
                step=0.5,
            )
            reserve = st.number_input(
                "Target stock reserve multiplier",
                min_value=1.0,
                max_value=3.0,
                value=1.15,
                step=0.05,
            )

            transfer_plan = plan_station_transfers(
                station_snapshot,
                forecast_horizon_hours=horizon,
                reserve_multiplier=reserve,
            )
            st.dataframe(transfer_plan, use_container_width=True)
            st.download_button(
                "Download station transfer shortlist",
                transfer_plan.to_csv(index=False).encode("utf-8"),
                file_name="station_transfer_shortlist.csv",
                mime="text/csv",
            )
            st.caption(
                "Validate expected returns, dock capacity, travel time and "
                "operational constraints before dispatch."
            )
        except Exception as exc:
            st.error(f"Could not create transfer shortlist: {exc}")
    else:
        st.info(
            "No station-level file uploaded. The historical dataset cannot "
            "identify donor or receiver stations."
        )

    st.divider()
    st.subheader("5. Proposed 4–6 week controlled redistribution pilot")
    st.markdown(
        """
        **Treatment:** forecast-guided redistribution in assigned station clusters.  
        **Control:** existing redistribution policy in matched clusters.

        - Week 1: validate station inventory, failed-rental and driver logs; collect baseline.
        - Weeks 2–4 (four-week pilot) or weeks 2–6 (six-week pilot): run treatment and control concurrently.
        - Primary metric: empty-station minutes per station-hour during predefined peak windows.
        - Secondary metrics: failed rental attempts, completed rentals, idle-bike hours and cost per additional successful rental.
        - Guardrails: driver hours, kilometres, safety, bike damage and control-zone service quality.
        - Scale only if the primary outcome improves by a pre-agreed meaningful amount without unacceptable cost or guardrail deterioration.
        """
    )
    st.caption(
        "Use matched station clusters, randomise within pairs, log cross-zone "
        "transfers and pre-register the primary metric and decision threshold."
    )


# ============================================================
# 11. BUSINESS ASSISTANT
# ============================================================

elif page == "Business Assistant":
    st.header("Business Intelligence Assistant")
    st.write(
        "Explore historical demand, customer segments, weather patterns and "
        "fleet-planning implications."
    )

    with st.expander("Example questions"):
        st.markdown(
            """
            1. When is demand highest on working days?
            2. Compare working days and non-working days.
            3. Which customer segment contributes more?
            4. How does weather relate to demand?
            5. Should we buy more bikes or redistribute?
            """
        )

    question = st.text_input(
        "Ask your business question",
        placeholder="Example: When is demand highest on working days?",
    )

    if st.button("Analyze Question"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            q = question.lower().strip()

            if any(term in q for term in ["weather", "rain", "humidity", "temperature"]):
                st.subheader("Weather and demand")
                weather_data = get_weather_summary(hour_df)
                weather_data.index = [
                    WEATHER_LABELS.get(int(code), f"Weather {code}")
                    for code in weather_data.index
                ]
                clear_mean = float(
                    hour_df.loc[hour_df["weathersit"] == 1, "cnt"].mean()
                )
                weather_data["Change vs clear (%)"] = (
                    weather_data["mean"] / clear_mean - 1
                ) * 100
                st.dataframe(weather_data.round(2), use_container_width=True)
                st.caption(
                    "These are historical associations, not proof of causation. "
                    "Weather category 4 has very few observations."
                )

            elif any(term in q for term in ["casual", "registered", "customer", "segment"]):
                st.subheader("Customer segment analysis")
                result = hour_df.groupby("workingday")[["casual", "registered"]].sum()
                result.index = ["Non-Working Day", "Working Day"]
                result["Total"] = result["casual"] + result["registered"]
                result["Casual share (%)"] = result["casual"] / result["Total"] * 100
                result["Registered share (%)"] = (
                    result["registered"] / result["Total"] * 100
                )
                st.dataframe(result.round(2), use_container_width=True)

            elif any(term in q for term in ["working", "weekday", "weekend", "non-working", "compare"]):
                st.subheader("Working-day versus non-working-day demand")
                comparison = hour_df.groupby("workingday")["cnt"].mean()
                non_working = float(comparison.loc[0])
                working = float(comparison.loc[1])

                c1, c2, c3 = st.columns(3)
                c1.metric("Working-day average", f"{working:.2f}")
                c2.metric("Non-working-day average", f"{non_working:.2f}")
                c3.metric(
                    "Difference (%)",
                    f"{(working / non_working - 1) * 100:.1f}%",
                )

                hourly_comparison = (
                    hour_df.groupby(["hr", "workingday"])["cnt"]
                    .mean()
                    .unstack()
                )
                hourly_comparison.columns = ["Non-Working Day", "Working Day"]
                hourly_comparison["Difference"] = (
                    hourly_comparison["Working Day"]
                    - hourly_comparison["Non-Working Day"]
                )
                st.dataframe(hourly_comparison.round(2), use_container_width=True)

            elif any(term in q for term in ["peak", "highest", "busiest", "maximum"]):
                st.subheader("Peak demand analysis")
                if "working" in q or "weekday" in q:
                    data = hour_df[hour_df["workingday"] == 1]
                    label = "Working Days"
                elif "weekend" in q or "non-working" in q:
                    data = hour_df[hour_df["workingday"] == 0]
                    label = "Non-Working Days"
                else:
                    data = hour_df
                    label = "All Days"

                profile = (
                    get_hourly_profile(data)
                    .dropna()
                    .sort_values(ascending=False)
                )
                peak_hour = int(profile.index[0])
                c1, c2 = st.columns(2)
                c1.metric("Highest-demand hour", format_hour(peak_hour))
                c2.metric("Average completed rentals", f"{profile.iloc[0]:.1f}")
                st.write(
                    f"For {label}, the highest average demand occurs at "
                    f"{format_hour(peak_hour)}."
                )
                st.dataframe(profile.head(5).rename("Average Rentals"))

            elif any(
                term in q
                for term in [
                    "fleet", "bikes required", "buy bikes", "purchase",
                    "redistribute", "allocation", "shortage", "recommend",
                ]
            ):
                st.subheader("Operational recommendation")
                timing = demand_timing_recommendations(hour_df)
                st.write(
                    "Historical demand varies by hour and day type. Test "
                    "forecast-guided redistribution before recommending broad "
                    "fleet expansion."
                )
                st.markdown("**Working-day high-demand hours**")
                st.dataframe(pd.DataFrame(timing["working_day_top_hours"]))
                st.markdown("**Non-working-day high-demand hours**")
                st.dataframe(pd.DataFrame(timing["non_working_day_top_hours"]))
                st.warning(
                    "The dataset contains citywide completed rentals only. It does "
                    "not contain station inventory, failed rental attempts or trip "
                    "durations. It cannot prove station shortages or the need to "
                    "purchase more bikes."
                )

            else:
                st.info(
                    "Try asking about peak demand, working days, customers, weather, "
                    "fleet size or redistribution recommendations."
                )


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()
st.caption(
    "Bike Sharing Demand Intelligence | Historical Demand Analytics | "
    "Random Forest Forecasting | Operational Decision Support"
)
