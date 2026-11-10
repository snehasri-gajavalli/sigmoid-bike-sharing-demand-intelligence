# Decision Log — Bike Sharing Demand Intelligence

**Case study:** Sigmoid Data Science Trainee — Round 2  
**Dataset:** UCI Bike Sharing Dataset (Capital Bikeshare, 2011–2012)  
**Purpose:** Record the main analytical, modelling, prototype and business decisions and explain why they were made.

## 1. Business problem framing

**Decision:** Investigate whether better demand understanding, forecasting, fleet-health checks and redistribution should be tested before recommending broad fleet purchases.

**Reasoning:** The case describes riders reporting bike unavailability while bikes are observed sitting unused elsewhere. This is consistent with possible temporal or spatial mismatch, but it is not sufficient evidence to diagnose the cause.

**Limitation:** The supplied data records completed citywide rentals. It does not identify failed requests, station-level stock, idle locations or maintenance downtime. Redistribution is therefore a hypothesis to test, not an established root cause.

## 2. Dataset selection and quality checks

**Decision:** Use both `hour.csv` (17,379 rows) and `day.csv` (731 rows).

**Reasoning:** Hourly data supports intraday profiles and forecasting; daily data supports daily summaries and quality checks.

**Checks recorded:**
- Dataset dimensions, column types, recorded-row nulls and duplicates.
- Duplicate date-hour keys and timestamp continuity.
- Consistency of `casual + registered = cnt`.
- Zero humidity/windspeed readings and rare weather category 4.
- High-demand observations were inspected and retained unless independent evidence identified an error.

**Important treatment:** The timestamp audit identified 165 missing hourly timestamps across 76 incomplete days. These are absent timestamps, not null values in existing rows. They were not automatically filled with zero rentals. Complete days were used for the commute-share hypothesis test.

## 3. Feature engineering and leakage prevention

**Decision:** Use calendar, time and weather variables to estimate `cnt`.

**Predictors include:** season, year, month, hour, holiday, weekday, working-day indicator, weather situation, temperature, feels-like temperature, humidity and windspeed.

**Excluded:**
- `casual` and `registered`, because they sum directly to `cnt` and would leak the target.
- `instant`, because it is a row identifier rather than a meaningful demand feature.

**Reasoning:** Predictors should be information that could reasonably be available at prediction time. For live forecasting, future weather would need to come from a weather forecast source rather than from future observed weather.

## 4. Exploratory-analysis choices

**Decision:** Analyze demand by hour, working-day status, customer segment, weather and season instead of relying only on one overall average.

**Key recorded findings:**
- 3,292,679 hourly completed rentals; approximately 189.5 rentals per recorded hour.
- Maximum observed hour: 977 rentals.
- Overall highest average hour: 17:00 at 461.45 rentals/hour; 18:00 follows at 425.51.
- Working-day average: 193.21 rentals/hour; non-working-day average: 181.41.
- Registered share: 86.77% on working days; casual share: 31.66% on non-working days.
- Less favourable weather groups show lower descriptive averages, but this is observational, not causal. Weather category 4 has only three observations.

## 5. Primary hypothesis design

**Decision:** Test whether the share of daily rentals in commute windows is higher on working days than on non-working days.

**H0:** The commute-window share is the same across day types.  
**H1:** It is higher on working days.

**Window definition:** 07:00–09:00 and 17:00–19:00 (hours 7, 8, 9, 17, 18 and 19).

**Unit of analysis:** One complete day, not one hour. This avoids treating correlated hourly observations from the same day as independent replicates.

**Recorded result:** 443 complete working days and 212 complete non-working days; mean shares 51.63% and 26.25%; difference 25.39 percentage points. Mann–Whitney U = 93,434 (p = 1.71 × 10⁻⁹³); permutation p = 0.0002; bootstrap 95% CI = 24.67–26.06 percentage points; Cliff's delta = 0.9897.

**Interpretation:** Evidence supports a large difference in commute-window concentration in the analysed historical data. Consecutive daily observations may be temporally dependent. This result does not establish station-level movement or actual stock-outs.

## 6. Forecasting target and evaluation design

**Decision:** Predict hourly completed rentals (`cnt`) using a Random Forest Regressor.

**Evaluation:** Chronological 70% / 15% / 15% train-validation-test partition. Training uses 12,165 rows, validation 2,607 and test 2,607. The latest test period is kept separate from model selection.

**Recorded final test metrics:** MAE = 64.09 rentals/hour; RMSE = 94.54 rentals/hour; R² = 0.8068.

**Reasoning:** MAE gives average absolute error; RMSE gives additional weight to large misses; R² summarizes explained variation on the test observations. These latest values supersede older model-run numbers and are the values used in the final project documentation.

**Feature-importance interpretation:** `atemp` is the largest individual encoded feature (14.83%), followed by `hr_17` (9.38%), `hum` (8.54%), `hr_18` (8.14%), `temp` (6.02%) and `hr_8` (5.90%). Importance is model reliance, not causality. One-hot encoded hour columns should not be read as the total importance of hour.

**Operational caution:** Peak-period error is higher than off-peak error, particularly in the evening, and forecasts can understate peak demand. Forecasts should be treated as planning support, not ground truth.

## 7. Fleet-sizing scenario choices

**Decision:** Use Little's Law to create an adjustable scenario rather than claim the dataset reveals actual fleet size.

**Reference demand:** Observed 2012 P95 = 662 rentals/hour.  
**Assumptions:** 20-minute trip duration, 2.0× spatial positioning buffer and 10% maintenance reserve.

**Calculation:** `ceil(662 × (20/60) × 2.0 ÷ (1 − 0.10)) = 491 bikes`.

**Sensitivity:** At 662 rentals/hour and 10% maintenance, the scenario ranges from 276 to 1,104 bikes when varying trip duration (15/20/30 minutes) and buffer (1.5×/2×/3×).

**Limitation:** This is not a measured fleet requirement. The data has no actual trip-duration, station inventory, fleet, dock, maintenance or spatial-flow records. The optional station-transfer shortlist is a heuristic; synthetic test data validates software behaviour only.

## 8. Business recommendation and proposed experiment

**Decision:** Do not recommend broad fleet expansion solely from citywide completed-rental counts. Instrument station operations and test forecast-guided redistribution against business-as-usual first.

**Pilot design:** 4–6 weeks, matched station clusters, stable treatment/control assignment and pre-registered success criteria. If there are too few independent clusters, consider a randomized switchback design by cluster-week.

**Primary outcome:** Empty-station minutes per station-hour during defined peak windows.

**Secondary outcomes:** Failed rental attempts per 1,000 attempts, completed rentals per station-hour, station-hours within target stock, idle-bike hours and complaints.

**Guardrails:** Driver hours, distance, costs, cost per additional successful rental, safety incidents, bike damage and control-zone service quality.

**Decision rule:** Consider targeted purchases only if reliable station-level evidence shows shortages persist after redistribution and fleet-health improvements, and the service/cost/safety case supports purchase.

## 9. Streamlit dashboard decisions

**Decision:** Retain the original `app.py` and use `app_GEMINI_NLP.py` as the Gemini-enabled entry point, avoiding destructive replacement of the original app.

**Dashboard sections:** Demand Overview; Hourly Analysis; Customer Segmentation; Demand Prediction; Fleet Planning & Experiment; Business Assistant.

**Model artifact:** The current prediction workflow uses `models/bike_demand_model_bundle.pkl`, which contains the current fitted model/preprocessing schema. The older `bike_demand_random_forest.pkl` is a legacy artifact and is not interchangeable with the current one-hot model bundle.

## 10. Gemini Business Assistant design

**Decision:** Add natural-language question understanding using the Google `google-genai` SDK and manual function calling.

**Design safeguards:**
- Gemini can select only from an explicit allow-list of analytical tools.
- Function names and arguments are validated locally before execution.
- Python functions perform the calculations against the local Pandas data or saved model.
- Gemini receives aggregate function results for explanation, not raw CSV files or arbitrary executable Python.
- A numeric-grounding check checks generated numbers against the tool output.
- If the API key is absent or Gemini is unavailable, deterministic rule-based fallback remains available and the interface identifies fallback mode.
- The API key is read from an environment variable or Streamlit secrets; it is not written into source code.

**Current launch command:** `streamlit run app_GEMINI_NLP.py`.

**Live verification:** A real Gemini response was observed in the local dashboard for “When is demand highest?”. Gemini selected `get_hourly_demand_profile`, identified 17:00 and 461.45 completed rentals/hour, and passed the grounding check. Working/non-working comparison and customer-segment questions were also manually tried. The automated assistant tests are offline and do not call the live Gemini API.

## 11. Automated testing and repository decisions

**Test command:** `python -m pytest tests -v`  
**Recorded result:** 37 passed, 0 failed, 1 warning (27 project tests + 10 Business Assistant tests), local run on 10 November 2026. The warning did not cause test failure.

**Repository safety:** `.gitignore` excludes `.env`, `.streamlit/secrets.toml`, `.streamlit/credentials.toml`, virtual environments, caches and local `.vscode` settings. The legacy model is excluded. `models/bike_demand_model_bundle.pkl` is tracked using Git LFS because it exceeds the regular GitHub 100 MiB file limit.

## 12. Final decision

Prioritize station-level measurement, fleet-health improvements and a controlled forecast-guided redistribution pilot before broad fleet expansion. The historical dataset supports demand-timing analysis and completed-rental forecasting; it does not prove station-level shortages, causal redistribution effects or the exact fleet required.
