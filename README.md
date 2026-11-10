# Bike Sharing Demand Intelligence

**Sigmoid Data Science Trainee Case Study | **  
**Domain:** Data Science, Demand Forecasting and Business Analytics  
**Dataset:** UCI Bike Sharing Dataset — Capital Bikeshare, Washington, D.C. (2011–2012)

## Project overview

This project investigates whether a bike-sharing operator should immediately purchase more bikes or first use demand analysis, forecasting, fleet-health checks and demand-informed redistribution to improve service. It combines data quality checks, exploratory analysis, statistical testing, a Random Forest demand model, assumption-based fleet scenarios and a Streamlit decision-support application.

The key evidence boundary is important: the supplied data records **completed citywide rentals**. It does not record failed rental attempts, station-level stock, trip duration, station-to-station flows or maintenance downtime. The project can identify when completed rentals are concentrated and estimate completed-rental demand; it cannot prove where riders were unable to find bikes or establish the actual fleet required.

## Case-study requirements addressed

1. **Quality checks and features:** audited hourly and daily data, checked consistency and timestamp continuity, examined suspicious zero weather readings and created calendar/time/weather features for analysis and modelling.
2. **Testable hypothesis:** compared the share of daily rentals occurring in defined commute windows on working and non-working days, using complete days and day-level statistical tests.
3. **Demand model and fleet estimate:** trained and evaluated a chronological Random Forest model for hourly completed rentals, then used Little's Law with explicit assumptions to create scenario fleet estimates and sensitivity analysis.
4. **Natural-language prototype:** built a Gemini-powered Business Assistant in Streamlit. Gemini selects from approved analytical functions; local Python code validates and executes the functions against the project data/model and returns aggregate results for explanation.

## Dataset

| File | Records | Grain | Target / use |
|---|---:|---|---|
| `data/hour.csv` | 17,379 | Hourly | `cnt`: completed rentals per hour |
| `data/day.csv` | 731 | Daily | Daily completed rentals and daily aggregates |

Important fields include date, hour, season, year, month, weekday, holiday, working-day status, weather situation, normalized temperature (`temp`), feels-like temperature (`atemp`), humidity (`hum`), windspeed, casual rentals, registered rentals and total rentals (`cnt`).

## Main analytical findings

- Total recorded hourly rentals: **3,292,679**; average: approximately **189.5 per recorded hour**; maximum observed hour: **977 rentals**.
- The highest overall average hour is **17:00 (461.45 rentals/hour)**, followed by **18:00 (425.51)**.
- Working days average **193.21 rentals/hour** compared with **181.41** on non-working days (about **6.51% higher**). Working days show stronger commute peaks; non-working days are relatively more concentrated around midday.
- Registered customers account for **86.77%** of working-day rentals; casual customers account for **31.66%** of non-working-day rentals.
- Average completed rentals are lower in less favourable weather categories in the descriptive analysis. This is an observational association, not proof of causation. Weather category 4 has only three observations and is too sparse for a reliable standalone conclusion.

## Data quality and preparation

- `hour.csv` contains 17,379 recorded observations; `day.csv` contains 731 records.
- The timestamp audit found **165 missing hourly timestamps across 76 incomplete days**. These are absent observations, not null values in existing rows, and were not automatically converted into zero rentals.
- Recorded-row null checks, duplicate-row checks and duplicate date-hour key checks were reported as zero.
- The relationship `casual + registered = cnt` was checked.
- High rental counts were retained unless independently identified as errors; genuine peaks are important to the operating question.
- Zero humidity and windspeed values were flagged for interpretation rather than blindly imputed. Weather category 4 contains only three rows.
- `casual` and `registered` were excluded from model predictors because they sum to `cnt` and would leak the target. `instant` was excluded because it is an identifier.

## Hypothesis test

**H0:** The share of a day's rentals in the commute windows is the same on working and non-working days.  
**H1:** The share is higher on working days.

Commute windows are **07:00–09:00 and 17:00–19:00** (hours 7, 8, 9, 17, 18 and 19). Each complete day is one analysis unit; incomplete days are excluded to avoid comparing partial daily totals.

| Measure | Working days | Non-working days |
|---|---:|---:|
| Complete days | 443 | 212 |
| Mean commute-window share | 51.63% | 26.25% |
| Median commute-window share | 52.38% | 26.31% |

The mean-share difference is **25.39 percentage points**. Reported tests: Mann–Whitney U = **93,434**, p = **1.71 × 10⁻⁹³**; permutation p = **0.0002**; bootstrap 95% CI for the difference = **24.67–26.06 percentage points**; Cliff's delta = **0.9897**. This is strong evidence of different temporal concentration in this historical sample. Consecutive days may be temporally dependent, so inferential uncertainty should be interpreted with that caveat. It does not establish station-to-station movement or prove stock-outs.

Other analyses examined AM/PM/midday periods, weather groups and seasonal groups using non-parametric methods and adjusted pairwise comparisons. The selected high-temperature threshold comparison was not statistically significant in the final report; that does not establish that temperature has no relationship with demand.

## Demand forecasting model

**Target:** hourly completed rentals (`cnt`).  
**Model:** Random Forest Regressor with encoded categorical features and calendar, hour and weather/temperature predictors.  
**Evaluation:** chronological 70% / 15% / 15% train-validation-test split; the final test period was not used for model selection.

| Partition | Rows | Date range |
|---|---:|---|
| Train | 12,165 | 2011-01-01 to 2012-05-27 |
| Validation | 2,607 | 2012-05-27 to 2012-09-12 |
| Test | 2,607 | 2012-09-12 to 2012-12-31 |

Latest recorded test performance:

| Model | MAE (rentals/hour) | RMSE (rentals/hour) | R² |
|---|---:|---:|---:|
| Random Forest | 64.09 | 94.54 | 0.8068 |

The most important individual encoded features in the recorded model output include `atemp` (14.83%), `hr_17` (9.38%), `hum` (8.54%), `hr_18` (8.14%), `temp` (6.02%) and `hr_8` (5.90%). These are model importances, not causal effects. Because hour is one-hot encoded, an individual hour dummy is not the total importance of the hour variable.

Peak-period errors were recalculated from the final held-out predictions for the report. The final report defines the exact peak windows and bias as **prediction minus actual**; negative bias means underprediction. Peak errors are higher than off-peak errors, especially in the evening. This should be considered before using the model to schedule operations.

## Fleet scenario analysis

Little's Law is used as a transparent scenario framework:

`bikes in motion = hourly rental-start rate × average trip duration (hours)`

The illustrative scenario uses observed 2012 P95 demand of **662 rentals/hour**, assumed trip duration of **20 minutes**, assumed spatial positioning buffer of **2.0×**, and assumed maintenance reserve of **10%**:

`ceil(662 × (20/60) × 2.0 ÷ (1 − 0.10)) = 491 bikes`

This is an **assumption-based scenario, not a measured fleet requirement**. Sensitivity at 662 rentals/hour and 10% maintenance:

| Assumed trip duration | 1.5× buffer | 2.0× buffer | 3.0× buffer |
|---|---:|---:|---:|
| 15 minutes | 276 | 368 | 552 |
| 20 minutes | 368 | 491 | 736 |
| 30 minutes | 552 | 736 | 1,104 |

The dataset does not contain actual trip duration, station availability, fleet size or spatial distribution. The optional station-transfer shortlist is a heuristic; its synthetic-data test validates code behaviour, not real-world savings.

## Business recommendation and proposed pilot

Do **not** recommend broad fleet expansion using citywide completed-rental counts alone. First instrument station availability and fleet health, then run a **4–6 week matched-cluster redistribution pilot** against business-as-usual operations.

- Match comparable station clusters on baseline demand, geography, operating hours and stock-out measures; randomize within matched pairs where feasible.
- Pre-register peak windows, exclusions, primary outcome and decision threshold.
- Primary outcome: **empty-station minutes per station-hour** during pre-defined peak windows.
- Secondary outcomes: failed rental attempts per 1,000 attempts, completed rentals per station-hour, station-hours within target stock, idle-bike hours and complaints.
- Guardrails: driver hours, kilometres, operating cost, cost per additional successful rental, safety incidents, bike damage and control-zone service quality.
- Consider targeted purchases only if verified unmet demand persists after rebalancing and fleet-health improvements and the cost/safety case is acceptable.

Minimum additional operational data: timestamped station inventory and empty docks; trip start/end station and timestamps; failed/abandoned attempts and reasons; bike health/maintenance status; and rebalancing movements, driver hours, distance and cost.

## Gemini natural-language Business Assistant

The Gemini-enabled entry point is **`app_GEMINI_NLP.py`**. The original `app.py` is retained separately. The Business Assistant uses Gemini natural-language understanding and **manual function calling**:

1. A manager enters a plain-English question.
2. Gemini selects from an explicit allow-list of analytical functions.
3. The local dispatcher validates the function name and arguments.
4. The approved Python function runs against the local Pandas data or saved prediction model.
5. Aggregate function results are returned to Gemini for a concise explanation; Gemini does not receive raw CSV files or execute arbitrary Python.
6. A numeric-grounding check checks the generated answer against the returned result.
7. If the API key is missing or Gemini fails, deterministic rule-based fallback remains available and the interface identifies the fallback mode.

The available function categories cover hourly demand, day-type comparison, customer segmentation, weather summary, demand prediction, fleet scenarios, redistribution guidance and data limitations. The separate Demand Prediction page remains available for structured inputs.

**Live verification:** a real Gemini response was observed in the local Streamlit app for “When is demand highest?”. The UI showed Gemini NLP + manual function calling, selected `get_hourly_demand_profile`, returned the 17:00 peak (461.45 rentals/hour) and passed the numeric grounding check. Day-type and customer-segment questions were also manually tried. The automated Business Assistant tests are offline tests; they do not call the live Gemini service.

## Validation

The combined local test command was:

```powershell
python -m pytest tests -v
```

Recorded result on 10 November 2026: **37 passed, 0 failed, 1 warning** (27 existing project tests plus 10 Business Assistant tests). The warning did not cause a test failure. Live Gemini API execution was verified separately in Streamlit.

## Run locally

1. Install Python and Git LFS (the current model bundle is stored with Git LFS).
2. Clone the repository and fetch LFS assets:

```powershell
git lfs install
git clone <repository-url>
cd sigmoid-bike-sharing-demand-intelligence
```

3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Set your Gemini API key in the same PowerShell session used to launch the app:

```powershell
$env:GEMINI_API_KEY = "YOUR_API_KEY"
```

5. Launch the Gemini-enabled app:

```powershell
streamlit run app_GEMINI_NLP.py
```

To run tests:

```powershell
python -m pytest tests -v
```

Do not paste the API key into source files, notebooks, screenshots or GitHub. The `.gitignore` excludes `.env`, `.streamlit/secrets.toml`, credentials and virtual environments. Keep actual secrets out of the repository.

## Repository structure

```text
sigmoid-bike-sharing-demand-intelligence/
├── data/                         # hour.csv, day.csv
├── models/                       # current model bundle (Git LFS)
├── notebooks/                    # four analysis/forecasting notebooks
├── outputs/                      # evaluation CSVs and architecture figure
├── tests/                         # 27 project + 10 assistant tests
├── app.py                         # original app retained
├── app_GEMINI_NLP.py              # Gemini-enabled Streamlit entry point
├── business_assistant_nlp.py      # Gemini schemas, dispatcher and tools
├── fleet_planning.py              # scenario and planning functions
├── generate_prototype_architecture.py
├── redistribution_experiment_plan.md
├── requirements.txt
├── README.md
├── IMPLEMENTATION_README.md
├── Final_Case_Study_Report.md
├── DECISION_LOG.md
├── AI_USAGE.md
└── .gitignore
```

## Limitations and future work

- The model predicts completed rentals, not suppressed demand or station stock-outs.
- Citywide hourly data cannot identify which station needs bikes or which station has surplus bikes.
- Fleet calculations depend on assumed trip duration, spatial buffer and maintenance reserve.
- Weather category 4 has only three rows; it should not support a strong standalone conclusion.
- Consecutive daily observations may be temporally dependent.
- Future weather must be available at prediction time in a live forecasting workflow; evaluation should avoid using information unavailable when decisions are made.
- The station-transfer shortlist is a heuristic, not a route optimizer, and synthetic test inputs do not prove savings.

Future work should add station-level inventory, trips, failed rental logs, bike health and rebalancing data; develop station-level forecasts; optimize redistribution under route and capacity constraints; and evaluate the strategy with a controlled operational pilot.
