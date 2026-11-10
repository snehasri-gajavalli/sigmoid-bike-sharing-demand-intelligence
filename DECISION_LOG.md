# Decision Log — Bike Sharing Demand Intelligence

**Case:** Sigmoid Data Science Trainee — Round 2  
**Dataset:** UCI Bike Sharing Dataset (Capital Bikeshare, Washington, D.C., 2011–2012)  
**Purpose:** Record the key analytical, modelling, prototype and business decisions.

## 1. Business problem

**Decision:** Investigate demand patterns, forecasting, fleet health and redistribution before recommending broad fleet purchases.

**Why:** Reports of unavailable bikes alongside apparently unused bikes elsewhere may indicate temporal or spatial mismatch, but this is only a hypothesis. The dataset contains completed citywide rentals, not failed requests, station inventory, bike locations or maintenance downtime.

## 2. Data and quality

**Decision:** Use both `hour.csv` (17,379 rows) and `day.csv` (731 rows). Hourly data supports intraday analysis and forecasting; daily data supports daily summaries and checks.

**Checks and treatment:**
- Checked dimensions, data types, row nulls, duplicates, date-hour keys and timestamp continuity.
- Verified `casual + registered = cnt`.
- Found 165 missing hourly timestamps across 76 incomplete days. These are absent observations, not nulls in existing rows; they were not filled with zero.
- Investigated zero humidity/windspeed and rare weather category 4 (only 3 observations).
- Retained high-demand observations unless independent evidence showed an error.
- Used complete days for the commute-share hypothesis test.

## 3. Features and leakage prevention

**Decision:** Predict hourly completed rentals (`cnt`) using calendar, time and weather information: season, year, month, hour, holiday, weekday, working-day status, weather situation, temperature, feels-like temperature, humidity and windspeed.

**Excluded:** `casual` and `registered` because they sum directly to `cnt` (target leakage); `instant` because it is only a row identifier.

**Operational note:** A live forecast must use weather forecasts available at prediction time, not future observed weather.

## 4. EDA decisions and findings

**Decision:** Compare demand by hour, working-day status, customer segment, weather and season rather than relying only on an overall average.

**Key findings:**
- 3,292,679 recorded hourly rentals; average 189.5 rentals per recorded hour; maximum observed hour 977.
- Highest average hour: 17:00 (461.45 rentals/hour), followed by 18:00 (425.51).
- Working-day average: 193.21 rentals/hour; non-working-day average: 181.41 (about 6.51% lower).
- Working days show stronger 07:00–09:00 and 17:00–19:00 peaks; non-working-day activity is relatively more concentrated around 12:00–15:00.
- Registered share: 86.77% on working days; casual share: 31.66% on non-working days.
- Less favourable weather groups have lower descriptive averages. This is observational, not causal; weather category 4 has only three observations.

## 5. Commute-share hypothesis

**Decision:** Test whether the share of daily rentals in commute windows is higher on working days.

- **H0:** Commute-window share is the same across day types.
- **H1:** Commute-window share is higher on working days.
- Windows: 07:00–09:00 and 17:00–19:00 (hours 7, 8, 9, 17, 18, 19).
- Unit: one complete day, not one hour, to avoid treating within-day hourly observations as independent.

**Results:** 443 complete working days and 212 complete non-working days; mean shares 51.63% and 26.25%; difference 25.39 percentage points. Mann–Whitney U = 93,434 (p = 1.71 × 10⁻⁹³); permutation p = 0.0002; bootstrap 95% CI = 24.67–26.06 percentage points; Cliff's delta = 0.9897.

**Interpretation:** Strong evidence of a large historical difference in commute-window concentration. Consecutive days may be temporally dependent. This does not prove station-level stock-outs or origin-destination flows.

## 6. Forecasting decisions

**Decision:** Predict hourly completed rentals (`cnt`) with a Random Forest Regressor.

**Evaluation:** Chronological 70% / 15% / 15% train-validation-test split:
- Train: 12,165 rows
- Validation: 2,607 rows
- Test: 2,607 rows

The latest test period was held out from model selection.

**Final test metrics:** MAE = 64.09 rentals/hour; RMSE = 94.54 rentals/hour; R² = 0.8068. These latest results supersede older model-run values.

**Feature importance:** `atemp` 14.83%, `hr_17` 9.38%, `hum` 8.54%, `hr_18` 8.14%, `temp` 6.02%, `hr_8` 5.90%. Importance indicates model reliance, not causation; one-hot hour columns do not represent the total importance of the hour variable.

**Operational caution:** Peak-period errors are higher than off-peak errors, especially in the evening, and predictions can understate peak demand. Treat forecasts as planning support, not ground truth.

## 7. Fleet-sizing scenario

**Decision:** Use Little's Law for an adjustable scenario, not as a claim about actual fleet size.

- Historical reference: 2012 P95 = 662 rentals/hour.
- Assumptions: 20-minute trip, 2.0× spatial positioning buffer, 10% maintenance reserve.
- Calculation: `ceil(662 × (20/60) × 2.0 ÷ (1 − 0.10)) = 491 bikes`.
- Sensitivity across trip duration (15/20/30 minutes) and buffer (1.5×/2×/3×): 276–1,104 bikes.

This is not a measured fleet requirement. Actual requirements need trip duration, station inventory, fleet, docks, maintenance and spatial-flow data. The station-transfer shortlist is a heuristic; synthetic data tests software behaviour, not real savings.

## 8. Recommendation and pilot

**Decision:** Do not recommend broad fleet expansion from citywide completed-rental counts alone. Instrument station operations and test forecast-guided redistribution against business-as-usual.

**Pilot:** 4–6 weeks using matched station clusters, stable treatment/control assignment and pre-registered success criteria. Consider a randomized cluster-week switchback only if there are too few independent clusters.

- **Primary outcome:** Empty-station minutes per station-hour during defined peak windows.
- **Secondary outcomes:** Failed rental attempts per 1,000 attempts, completed rentals per station-hour, target-stock availability, idle-bike hours and complaints.
- **Guardrails:** Driver hours/distance, cost and cost per additional successful rental, safety incidents, bike damage and control-zone service quality.

Consider targeted purchases only if verified shortages persist after redistribution and fleet-health improvements.

## 9. Dashboard and Gemini assistant

**Decision:** Retain `app.py`; use `app_GEMINI_NLP.py` as the Gemini-enabled entry point.

**Dashboard areas:** Demand Overview; Hourly Analysis; Customer Segmentation; Demand Prediction; Fleet Planning & Experiment; Business Assistant.

**Model artifact:** Current workflow uses `models/bike_demand_model_bundle.pkl` (model plus preprocessing schema). `bike_demand_random_forest.pkl` is a legacy artifact and is not interchangeable.

**Assistant design:** Uses Google's `google-genai` SDK and manual function calling. Gemini selects only allow-listed tools; local code validates function names and arguments; Python functions calculate against local Pandas data or the saved model; only aggregate results are sent for explanation. Numeric grounding checks generated claims. A deterministic fallback is used and labelled if the API key is missing or Gemini is unavailable. The API key is stored in environment variables or Streamlit secrets, not source code.

**Launch:** `streamlit run app_GEMINI_NLP.py`

**Verification:** A live Gemini response to “When is demand highest?” selected `get_hourly_demand_profile`, returned 17:00 and 461.45 completed rentals/hour, and passed the grounding check. Day-type and customer-segment questions were also manually tried. Automated assistant tests are offline and do not call the live API.

## 10. Tests and repository safety

**Test command:** `python -m pytest tests -v`  
**Recorded result (10 November 2026):** 37 passed, 0 failed, 1 warning (27 project tests + 10 Business Assistant tests).

`.gitignore` excludes secrets, `.env`, virtual environments, caches and local editor settings. The legacy model is excluded. `models/bike_demand_model_bundle.pkl` is tracked with Git LFS because it exceeds GitHub's regular 100 MiB file limit.

## Final decision

Prioritize station-level measurement, fleet-health improvements and a controlled forecast-guided redistribution pilot before broad fleet expansion. The historical dataset supports demand-timing analysis and completed-rental forecasting; it does not prove station-level shortages, causal redistribution effects or the exact fleet required.
