# Sigmoid Bike-Sharing --- Fleet Planning and Pilot Deliverables

This update adds fleet-planning, redistribution-experiment and
dashboard-assistant functionality to the existing bike-sharing
case-study project. It does **not** create, remove or replace any of the
four completed notebooks.

## Files

-   `app.py` --- updated Streamlit dashboard.
-   `fleet_planning.py` --- helper functions for Little's Law fleet
    scenarios, observed demand percentiles, sensitivity analysis,
    demand-timing summaries and an optional station-transfer shortlist.
-   `redistribution_experiment_plan.md` --- proposed 4--6 week
    controlled redistribution pilot.

## Install into the existing project

Copy the updated files into the root of your existing
`Sigmoid_Bike_Sharing_Case_Study/` folder. Replace the existing `app.py`
and `fleet_planning.py` with the supplied updated versions; add
`redistribution_experiment_plan.md`.

Keep the existing `data/`, `models/`, `notebooks/`, `outputs/` and
`tests/` folders, along with the four completed notebooks.

The current forecasting workflow uses the model bundle:

-   `models/bike_demand_model_bundle.pkl`

The older artifact:

-   `models/bike_demand_random_forest.pkl`

is a legacy 12-feature model. Do not assume it is interchangeable with
the newer one-hot-encoded model bundle. Keep it if other project code
still references it, but use the artifact expected by the current
`app.py`.

Install dependencies from the project root if needed:

``` bash
pip install -r requirements.txt
```

If you are installing manually, the dashboard uses Streamlit, pandas,
NumPy, joblib and the plotting libraries imported by `app.py`. The
project requirements file is the preferred source of dependency
versions.

Run from the project root:

``` bash
streamlit run app.py
```

## What changed

### Fleet Planning & Experiment page

-   Uses Little's Law: bikes in motion = hourly rental starts × average
    trip duration in hours.
-   Applies explicit spatial-positioning and maintenance assumptions to
    produce scenario estimates.
-   Shows observed 2012 P90, P95, P99 and maximum hourly
    completed-rental counts.
-   Provides a sensitivity grid over trip duration, spatial buffer and
    maintenance reserve.
-   Summarises working-day and non-working-day demand timing.
-   Accepts a station snapshot CSV and produces a basic donor/receiver
    shortlist.

Required station snapshot CSV columns:

``` text
station_id
available_bikes
forecast_rentals_per_hour
```

The shortlist is a simple target-stock heuristic. It is not a route
optimiser or a validated rebalancing plan. Check any proposed transfers
against returns, dock capacity, travel time, driver capacity and bike
health before operational use.

### Business Assistant

Questions about recommendations, redistribution, allocation, bike
purchases or station shortages can surface:

-   Demand-timing evidence from the available dataset
-   A proposed 4--6 week matched-cluster pilot
-   Primary and secondary outcome metrics
-   Conditions under which targeted purchases could be considered
-   Limitations of the citywide dataset

## Important limitations

The UCI hourly dataset contains citywide completed rentals. It does not
provide actual trip duration, station-level inventory, station-level
flows or failed rental attempts.

Therefore:

-   Fleet sizes are assumption-based scenarios, not verified fleet
    requirements.
-   Citywide data cannot identify specific donor or receiver stations.
-   The station-transfer upload feature is useful only when supplied
    with genuine, current station-level operational data and forecasts.
-   The proposed experiment requires baseline operational
    instrumentation before a redistribution effect can be measured.
-   Completed rentals alone cannot establish how much demand was unmet
    because bikes were unavailable.

Treat the pilot as a way to test whether forecast-guided redistribution
improves service, not as evidence that redistribution has already been
proven to work.

## Validation status

The project's earlier local test run reported **27 passed, 0 failed and
0 skipped**, with one third-party `anyio` deprecation warning. The
Streamlit dashboard was also launched locally, and the Demand Prediction
page returned a prediction for a sample input.

The `fleet_planning.py` file was subsequently revised to strengthen
input validation and clarify the status of its timing and transfer
heuristics. After replacing the local file with that revised version,
rerun:

``` bash
python -m pytest tests/test_prototype.py -v
```

Then launch the dashboard:

``` bash
streamlit run app.py
```

A passing test suite and successful dashboard launch confirm software
behaviour for the tested inputs; they do not validate the fleet
assumptions or establish that a redistribution pilot will improve
real-world outcomes.
