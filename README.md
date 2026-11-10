# Bike Sharing Demand Intelligence

### Data Science Case Study -- Sigmoid

**Project domain:** Data Science \| Demand Forecasting \| Business
Analytics

## 1. Project Overview

A data-driven bike-sharing demand analysis and forecasting project
developed to investigate operational inefficiencies in bike-sharing
systems.

The project analyzes historical rental patterns, identifies demand
variations across time and weather conditions, develops a machine
learning model for hourly demand forecasting, and presents business
insights through an interactive Streamlit dashboard.

The primary business objective is to assess whether demand-informed
redistribution should be investigated before expanding the existing bike
fleet.

## 2. Business Problem

Bike-sharing operators may experience situations where customers cannot
find available bikes in some locations while bikes remain underutilized
elsewhere.

This project investigates:

-   When and under what conditions is bike demand highest?
-   How does demand differ between working and non-working days?
-   How do casual and registered customers behave differently?
-   How is rental demand associated with weather conditions?
-   Can hourly demand be forecast using machine learning?
-   What evidence would be needed to decide whether redistribution
    should be explored before purchasing additional bikes?

## 3. Dataset

**Source:** UCI Bike Sharing Dataset

  Dataset        Records Description
  ------------ --------- -----------------------
  `hour.csv`      17,379 Hourly rental records
  `day.csv`          731 Daily rental records

**Target variable:** `cnt` -- total number of completed bike rentals.

Important features include hour, season, working day, weather situation,
temperature, feels-like temperature, humidity and windspeed.

## 4. Project Workflow

### Data Understanding and Cleaning

-   Dataset structure and data types
-   Missing timestamp investigation
-   Duplicate and consistency checks
-   Outlier analysis
-   Feature engineering

### Exploratory Data Analysis

-   Hourly demand patterns
-   Working-day and non-working-day comparisons
-   Customer segmentation
-   Seasonal and weather analysis
-   Peak-demand identification

### Statistical Analysis

-   Mann--Whitney U tests
-   Kruskal--Wallis tests
-   Pairwise comparisons
-   Hypothesis testing and interpretation

### Machine Learning

-   Chronological data splitting
-   Random Forest demand forecasting
-   Model evaluation using MAE, RMSE and R²
-   Feature-importance analysis
-   Prediction error analysis

The final reported model metrics should be taken from the latest saved
evaluation output. Earlier baseline and Gradient Boosting results are
not reproduced here because they belong to an earlier evaluation run and
should not be compared with the latest Random Forest result unless all
models are re-evaluated on the same final split.

### Business Intelligence

-   Assumption-based fleet-sizing scenarios
-   Sensitivity analysis
-   Demand-timing summaries
-   Optional station-transfer shortlist using uploaded operational data
-   Redistribution pilot design
-   Interactive business assistant

## 5. Machine Learning Model

**Selected model:** Random Forest Regressor

The forecasting workflow uses calendar, time, weather and
temperature-related predictors to estimate hourly rental counts (`cnt`).
Categorical variables are encoded as required by the trained model
bundle.

### Latest reported model performance

  Model               MAE    RMSE       R²
  --------------- ------- ------- --------
  Random Forest     64.09   94.54   0.8068

The values above are from the latest supplied
`forecasting_test_metrics.csv`. They supersede the older Random Forest
results (MAE 45.78, RMSE 71.02, R² 0.8955) and should be used
consistently in the report and presentation.

The latest workflow uses a chronological 70% / 15% / 15%
train-validation-test split. Model selection should be based on
validation performance; the test set is reserved for final evaluation.

**Interpretation:** MAE represents the average absolute prediction error
in rentals per hour. RMSE penalizes larger errors more heavily. R²
describes the proportion of target variation explained by the model on
the evaluated test set.

## 6. Key Business Findings

The analysis indicates:

-   Working days exhibit stronger morning and evening demand peaks.
-   Non-working days show relatively stronger midday demand.
-   Registered customers contribute a larger share of working-day
    rentals.
-   Casual customers contribute a larger share of non-working-day
    rentals.
-   Adverse weather conditions are associated with lower average rental
    demand.
-   The latest model error analysis indicates that prediction errors are
    higher during important peak periods, particularly in the evening.

These are patterns and associations in the historical dataset; they do
not by themselves establish causal effects or prove that customers were
unable to rent bikes.

## 7. Interactive Dashboard

The Streamlit application provides six sections:

1.  **Demand Overview:** Overall rental statistics and demand patterns.
2.  **Hourly Analysis:** Demand variations across hours and day types.
3.  **Customer Segmentation:** Casual and registered customer
    comparisons.
4.  **Demand Prediction:** Hourly demand estimates using the current
    trained model bundle.
5.  **Fleet Planning & Experiment:** Fleet scenarios, sensitivity
    analysis, demand timing and an optional station-transfer shortlist.
6.  **Business Assistant:** Rule-based business insights,
    recommendations and pilot guidance.

## 8. Business Recommendations

Investigate demand-informed redistribution before committing to
additional fleet purchases.

Potential operational improvements include:

-   Identify high-demand time windows.
-   Develop different redistribution schedules for working and
    non-working days.
-   Instrument station-level availability and failed rental attempts.
-   Conduct a controlled redistribution pilot using matched treatment
    and control station clusters.
-   Measure unmet demand, fleet utilization, operating cost and service
    quality.
-   Consider targeted purchases only if verified shortages persist after
    operational improvements.

The citywide dataset alone cannot establish that specific stations
experience shortages or excess inventory. Station-level operational data
is required.

## 9. Project Structure

The project contains four completed notebooks. Keep the existing
notebook files and their actual filenames; the abbreviated tree below
does not attempt to rename or replace them.

``` text
Sigmoid_Bike_Sharing_Case_Study/
├── data/
│   ├── hour.csv
│   └── day.csv
├── notebooks/
│   └── [four completed analysis and forecasting notebooks]
├── models/
│   ├── bike_demand_model_bundle.pkl
│   └── bike_demand_random_forest.pkl  # legacy artifact; retain if referenced
├── outputs/
│   ├── forecasting_test_metrics.csv
│   └── [other analysis and evaluation outputs]
├── tests/
│   └── test_prototype.py
├── app.py
├── fleet_planning.py
├── redistribution_experiment_plan.md
├── requirements.txt
├── README.md
├── IMPLEMENTATION_README.md
├── Final_Case_Study_Report.md
├── DECISION_LOG.md
└── AI_USAGE.md
```

The current app's prediction workflow uses
`models/bike_demand_model_bundle.pkl`. The older
`bike_demand_random_forest.pkl` is a legacy 12-feature model and is not
interchangeable with the newer one-hot-encoded model bundle. Keep it if
other code still depends on it.

## 10. Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   SciPy
-   Joblib
-   Streamlit
-   Pytest
-   Jupyter Notebook

## 11. Installation and Execution

### Step 1: Open the project

Open the `Sigmoid_Bike_Sharing_Case_Study/` root directory in VS Code or
a terminal.

### Step 2: Create a virtual environment

``` bash
python -m venv venv
```

Activate it on Windows PowerShell:

``` powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

``` bash
python -m pip install -r requirements.txt
```

### Step 4: Run the dashboard

``` bash
streamlit run app.py
```

Run the command from the project root so that the relative data and
model paths resolve correctly.

## 12. Automated Testing

The test suite covers dataset and model artifacts, required model
features, prediction behaviour, fleet calculations, input validation,
demand percentiles, sensitivity analysis, timing recommendations and
station-transfer planning.

Run:

``` bash
python -m pytest tests/test_prototype.py -v
```

**Previously observed test result:** 27 passed, 0 failed and 0 skipped,
with one third-party `anyio` deprecation warning.

This result reflects the earlier local test run. Rerun the suite after
installing the latest revised `fleet_planning.py` to confirm the current
working copy.

## 13. Limitations

The UCI dataset contains citywide completed rental counts rather than
station-level inventory or unfulfilled requests.

It does not directly provide:

-   Station-level bike availability.
-   Failed rental attempts.
-   Actual trip duration.
-   Bike arrival and departure flows by station.
-   Fleet maintenance status.

Therefore, the analysis cannot independently establish that bikes were
unavailable to customers or idle at particular stations.

Fleet estimates are scenarios based on explicit assumptions, not
verified fleet requirements. The optional station-transfer shortlist
requires genuine station-level operational data and forecasts; it is a
heuristic, not a route optimiser.

## 14. Future Improvements

-   Integrate real-time station availability data.
-   Incorporate trip duration and destination information.
-   Develop station-level demand forecasting.
-   Optimise redistribution routes using operational constraints.
-   Track unmet demand and fleet utilization.
-   Extend the business assistant with an LLM-based interface, with
    suitable safeguards and source-grounded responses.

## 15. Conclusion

This project combines exploratory data analysis, statistical testing,
machine learning and business intelligence to understand bike-sharing
demand and support operational decision-making.

The latest Random Forest evaluation reports an R² of 0.8068, with an MAE
of 64.09 and RMSE of 94.54 rentals per hour.

The findings support investigating demand-informed redistribution and
collecting station-level availability data before committing to
additional fleet purchases. A controlled pilot is needed to determine
whether redistribution improves service in practice.

------------------------------------------------------------------------

**Dataset:** UCI Bike Sharing Dataset\
**Project:** Sigmoid Data Science Case Study\
**Application:** Streamlit Dashboard
