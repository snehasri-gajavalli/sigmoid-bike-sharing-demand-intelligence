# Bike Sharing Demand Intelligence
## Final Case Study Report — Sigmoid Data Science Trainee, Round 2

**Dataset:** UCI Bike Sharing Dataset — Capital Bikeshare, Washington, D.C. (2011–2012)  
**Project domain:** Demand analysis, forecasting, fleet planning and natural-language decision support  
**Prototype:** Gemini-enabled Streamlit Business Assistant

---

## Executive summary

This case study investigates a bike-sharing operator's question: riders say they cannot find bikes when needed, while bikes are seen unused elsewhere. Should the operator buy more bikes, or first improve allocation and operations?

The project analyzes 17,379 hourly records and 731 daily records. It identifies temporal differences between working and non-working days, customer-segment differences and weather associations. A Random Forest Regressor predicts hourly **completed rentals** with a latest recorded chronological test MAE of 64.09, RMSE of 94.54 and R² of 0.8068. A Little's Law scenario produces 491 bikes under a specific set of assumptions; this is not a measured fleet requirement. A Streamlit dashboard provides six decision-support areas, including a Gemini-powered natural-language Business Assistant using validated manual function calling.

**Recommendation:** Do not recommend broad fleet expansion based only on citywide completed-rental counts. First instrument station availability, failed rental attempts and fleet health, then test forecast-guided redistribution in a controlled pilot. Consider targeted purchases only if verified shortages persist after redistribution and repairs.

## 1. Business problem and analytical framing

The reported situation may reflect temporal or spatial imbalance, poorly timed redistribution, unavailable/damaged bikes, technology failures or insufficient capacity. The supplied data can test temporal demand patterns, but it cannot directly identify which explanation is responsible.

The analysis distinguishes:

1. **Observed completed rentals** — available in `hour.csv` and `day.csv`.
2. **Predicted completed rentals** — estimated by the trained Random Forest.
3. **Unobserved/suppressed demand** — not recorded in the supplied dataset.
4. **Scenario fleet estimates** — calculations using explicit assumptions.
5. **Actual fleet sufficiency and station availability** — not measurable without station-level operational data.

This distinction prevents predicted rentals or a scenario estimate from being presented as evidence of actual stock-outs.

## 2. Dataset and data quality

### 2.1 Dataset overview

| File | Rows | Description |
|---|---:|---|
| `hour.csv` | 17,379 | Hourly completed-rental records |
| `day.csv` | 731 | Daily completed-rental records |

The target variable is `cnt`. Predictors and analysis fields include calendar variables, hour, working-day status, weather situation, temperature, feels-like temperature, humidity and windspeed. `casual` and `registered` describe customer segments.

### 2.2 Quality checks

- Recorded-row null checks, duplicate-row checks and duplicate date-hour key checks were reported as zero.
- The timestamp-continuity audit found **165 missing hourly timestamps across 76 incomplete days**. These are absent timestamps, not null values in rows that exist.
- Missing timestamps were not automatically converted to zero rentals.
- The identity `casual + registered = cnt` was checked.
- High rental observations were retained unless independent evidence showed they were data errors.
- Zero humidity and windspeed readings were flagged for interpretation rather than blindly replaced.
- Weather category 4 has only three observations, so standalone conclusions about it are unreliable.

**Why this matters:** filling missing timestamps with zero could distort hourly/daily demand profiles. Removing genuine peak observations could also reduce the model's ability to learn high-demand periods.

### 2.3 Feature engineering and leakage prevention

Calendar and time features include season, year, month, weekday, holiday, working-day indicator and hour. Weather predictors include weather situation, temperature, feels-like temperature, humidity and windspeed. The target is hourly completed rentals (`cnt`).

`casual` and `registered` were excluded from model predictors because they sum directly to `cnt`, which would leak target information. `instant` was excluded because it is a row identifier, not a predictive operational feature.

## 3. Exploratory data analysis

### 3.1 Overall demand

The dataset contains **3,292,679 recorded hourly rentals**, averaging approximately **189.5 completed rentals per recorded hour**. The maximum observed hourly count is **977**. These values show considerable variation across hours and conditions.

### 3.2 Hourly demand profile

| Hour | Average completed rentals/hour |
|---|---:|
| 17:00 | 461.45 |
| 18:00 | 425.51 |

The overall mean profile peaks at 17:00, followed by 18:00. This is a citywide timing result; it does not identify which stations need bikes.

### 3.3 Working versus non-working days

| Day type | Average rentals/hour |
|---|---:|
| Working day | 193.21 |
| Non-working day | 181.41 |

Working-day average demand is approximately **6.51% higher**. Working days have stronger morning and evening commute peaks, especially around 07:00–09:00 and 17:00–19:00. Non-working-day demand is relatively more concentrated around midday (approximately 12:00–15:00).

### 3.4 Customer segmentation

| Day type | Casual share | Registered share |
|---|---:|---:|
| Working day | 13.23% | 86.77% |
| Non-working day | 31.66% | 68.34% |

Registered customers contribute a larger proportion of working-day rentals. Casual customers contribute a larger proportion of non-working-day rentals. Separate day-type profiles may therefore be useful for operational planning.

### 3.5 Weather associations

| Weather category | Average hourly rentals | Interpretation |
|---|---:|---|
| Clear | 204.87 | Highest average among reported groups |
| Mist / cloudy | 175.17 | Lower than clear conditions |
| Light rain / snow | 111.58 | Substantially lower average |
| Heavy rain / snow | 74.33 | Only three observations; highly uncertain |

Less favourable weather categories are associated with lower average completed rentals. This is observational and does not prove weather caused the change. The smallest category should not be over-interpreted.

## 4. Statistical analysis and hypothesis testing

### 4.1 Primary hypothesis: commute-window concentration

**H0:** The share of a day's rentals in commute windows is the same on working and non-working days.  
**H1:** The share is higher on working days.

Commute windows are defined as 07:00–09:00 and 17:00–19:00 (hours 7, 8, 9, 17, 18 and 19). The analysis unit is one **complete day**, not one hourly record, because hourly observations within the same day are correlated. Days without all 24 hourly observations are excluded from this test.

| Metric | Working days | Non-working days |
|---|---:|---:|
| Complete days | 443 | 212 |
| Mean commute-window share | 51.63% | 26.25% |
| Median commute-window share | 52.38% | 26.31% |

| Statistical result | Value |
|---|---:|
| Mean-share difference | 25.39 percentage points |
| Mann–Whitney U | 93,434 |
| Mann–Whitney p-value | 1.71 × 10⁻⁹³ |
| Permutation test p-value | 0.0002 |
| Bootstrap 95% CI for difference | 24.67–26.06 percentage points |
| Cliff's delta | 0.9897 |

The test supports a very large historical difference in commute-window concentration. The observations are consecutive days, so temporal dependence may affect inferential uncertainty. The result indicates when demand is concentrated; it does not establish residential-to-office flows or prove station-level shortages.

### 4.2 Other statistical comparisons

The statistical workflow also examines AM, PM and midday periods, weather groups and seasonal groups using non-parametric comparisons and adjusted pairwise tests. The high-temperature hypothesis at the selected threshold was not statistically significant in the final project analysis. This means that the proposed difference was not established under that threshold; it does not prove temperature has no relationship with demand. Weather category 4 has only three observations.

## 5. Hourly demand forecasting

### 5.1 Target and predictors

The model target is hourly completed rentals (`cnt`). Predictors include calendar, time and weather fields. Customer segment counts are excluded because they sum to the target.

### 5.2 Chronological evaluation

A chronological 70% / 15% / 15% train-validation-test split was used, preserving time order.

| Partition | Rows | Date range |
|---|---:|---|
| Training | 12,165 | 2011-01-01 to 2012-05-27 |
| Validation | 2,607 | 2012-05-27 to 2012-09-12 |
| Test | 2,607 | 2012-09-12 to 2012-12-31 |

The earliest observations are used for fitting, the next period for model selection and the latest period for final testing. The test period is not used to choose the model.

### 5.3 Final model performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 64.09 | 94.54 | 0.8068 |

MAE is the average absolute prediction error in rentals/hour. RMSE penalizes larger misses more heavily. R² describes the proportion of target variation explained on the test observations. These latest recorded values supersede earlier model-run metrics; earlier runs are not mixed into this final result table.

### 5.4 Feature importance

The highest individual encoded importances in the recorded output include:

| Encoded feature | Importance |
|---|---:|
| `atemp` | 14.83% |
| `hr_17` | 9.38% |
| `hum` | 8.54% |
| `hr_18` | 8.14% |
| `temp` | 6.02% |
| `hr_8` | 5.90% |
| `workingday_1` | 4.38% |
| `workingday_0` | 4.15% |
| `hr_19` | 2.90% |

`atemp` is the strongest individual encoded feature. Because categorical variables are one-hot encoded, an individual hour dummy is not the total importance of the hour variable. Feature importance describes model reliance, not causation.

### 5.5 Error analysis

The final report uses held-out test predictions to compare morning peak, evening peak and off-peak errors. The most important pattern is that peak-period errors exceed off-peak errors, with the evening peak the most difficult period. Prediction bias is defined as **predicted demand minus actual demand**; a negative value indicates underprediction. The report's peak-error chart and table are based on the final held-out predictions and define the corresponding periods explicitly.

The model predicts completed rentals based on historical patterns; it does not estimate latent demand, unserved requests or station-specific stock-outs. For live future forecasts, weather information must be available at the decision time.

## 6. Fleet planning and scenario analysis

### 6.1 Little's Law

Little's Law relates the average number of bikes in motion to rental-start rate and trip duration:

`L = λ × W`

Here, `L` is average bikes in use, `λ` is hourly rental starts and `W` is average trip duration in hours. A spatial positioning buffer and maintenance reserve are additional planning assumptions.

### 6.2 Illustrative scenario

| Input/result | Value | Status |
|---|---:|---|
| Observed hourly demand reference (2012 P95) | 662 rentals/hour | Historical reference |
| Average trip duration | 20 minutes | Assumption |
| Spatial positioning buffer | 2.0× | Assumption |
| Maintenance reserve | 10% | Assumption |
| Bikes in motion | 220.7 | Calculated scenario |
| Positioned bikes before reserve | 441.3 | Calculated scenario |
| Scenario fleet after reserve | 491 bikes | Calculated, rounded up |

`ceil(662 × (20/60) × 2.0 ÷ (1 − 0.10)) = 491 bikes`

This is **not a measured fleet requirement**. Actual fleet sufficiency depends on station-level availability, trip duration, returns, docks, maintenance and spatial distribution.

### 6.3 Sensitivity

At 662 rentals/hour and 10% maintenance:

| Trip duration | 1.5× buffer | 2.0× buffer | 3.0× buffer |
|---|---:|---:|---:|
| 15 minutes | 276 | 368 | 552 |
| 20 minutes | 368 | 491 | 736 |
| 30 minutes | 552 | 736 | 1,104 |

The optional station-transfer shortlist is a heuristic that requires station data. Synthetic test data validates its software path, not real operating savings.

## 7. Streamlit dashboard and Gemini natural-language prototype

### 7.1 Dashboard

The Streamlit application contains six areas:

1. **Demand Overview** — overall rental statistics and patterns.
2. **Hourly Analysis** — demand by hour and day type.
3. **Customer Segmentation** — casual and registered customer comparisons.
4. **Demand Prediction** — hourly completed-rental estimates using the current saved model bundle.
5. **Fleet Planning & Experiment** — demand percentiles, Little's Law scenarios, sensitivity, demand timing and optional station-transfer heuristic.
6. **Business Assistant** — natural-language questions, evidence-backed answers and explicit data limitations.

### 7.2 Gemini function-calling workflow

The Gemini-enabled Streamlit entry point is `app_GEMINI_NLP.py`; the original `app.py` is retained separately. The assistant uses the Google `google-genai` SDK with manual function calling:

1. The operations manager enters a plain-English question.
2. Gemini interprets the question and selects from an allow-listed set of analytical tools.
3. The local dispatcher validates the function name and arguments.
4. Approved Python functions run against the local Pandas data or current saved model.
5. Aggregate function results are returned to Gemini for a manager-friendly explanation.
6. A numeric-grounding check verifies that generated numeric claims are supported by the returned result.
7. If the API key is absent or a Gemini request fails, deterministic rule-based fallback is used and the UI identifies the fallback mode.

The tool set covers hourly demand profiles, working/non-working day comparisons, customer segments, weather summaries, demand prediction, fleet scenarios, redistribution recommendations and data limitations. Gemini does not execute arbitrary Python code.

### 7.3 Prototype verification

A live Gemini response was observed in the local Streamlit app for the question “When is demand highest?”. The interface displayed “Answered using Gemini natural-language understanding and function calling”; the selected analytical function was `get_hourly_demand_profile`; the result identified 17:00 at 461.45 completed rentals/hour; and the numeric grounding check passed. Day-type comparison and customer-segment questions were also manually tried.

The automated Business Assistant tests are offline tests. They verify local functions, argument validation, prediction interface and fallback behaviour; they do not call the live Gemini API.

## 8. Automated testing

The complete local test command is:

```powershell
python -m pytest tests -v
```

Recorded result on 10 November 2026: **37 passed, 0 failed, 1 warning**. This includes 27 existing project tests and 10 Business Assistant tests. The warning did not cause a test failure. Live Gemini execution was verified separately in the running app.

## 9. Final recommendation and proposed experiment

The evidence supports a cautious operational sequence rather than immediate broad fleet expansion:

1. Instrument station inventory, failed rental attempts, trip records and bike health.
2. Use day-type-specific demand profiles to identify time windows for station-level forecasting and operational planning.
3. Run a matched-cluster forecast-guided redistribution pilot against business-as-usual for 4–6 weeks.
4. Pre-register the primary outcome, peak windows, exclusions, guardrails and decision threshold.
5. Consider targeted purchases only if verified unmet demand persists after redistribution and fleet-health improvements.

**Primary pilot metric:** empty-station minutes per station-hour during pre-defined peak windows.  
**Secondary metrics:** failed rental attempts per 1,000 attempts, completed rentals per station-hour, target-stock availability, idle-bike hours and complaints.  
**Guardrails:** driver hours, distance, costs, safety incidents, bike damage and control-zone service quality.

The dataset alone cannot validate the pilot effect because it lacks station-level operating data. The pilot must collect that information prospectively.

## 10. Limitations and next data requirements

- No station-level rentable-bike or empty-dock inventory.
- No failed or abandoned rental attempts and reason codes.
- No trip-level start/end stations or actual trip duration.
- No bike health or maintenance downtime records.
- No rebalancing movements, driver hours, distance or cost records.
- The model predicts completed rentals, not suppressed demand or actual shortages.
- Fleet scenarios depend on assumed trip duration, positioning buffer and maintenance reserve.
- The station-transfer shortlist is heuristic, not a route optimizer.
- Weather category 4 has only three observations; temporal dependence may affect statistical uncertainty.

Minimum additional data should include timestamped station inventory snapshots, trip start/end records, failed rental logs, bike health, and rebalancing movement/cost records.

## 11. Project deliverables

- Four Jupyter notebooks covering data understanding, EDA, hypothesis testing and demand forecasting.
- `app_GEMINI_NLP.py` — Gemini-enabled Streamlit entry point.
- `app.py` — retained original Streamlit app.
- `business_assistant_nlp.py` — Gemini function schemas, allow-listed dispatcher, validation, grounding and fallback.
- `fleet_planning.py` — fleet scenarios, percentiles, sensitivity and demand-timing functions.
- `data/` — hourly and daily dataset files.
- `models/bike_demand_model_bundle.pkl` — current model bundle and preprocessing schema (Git LFS).
- `outputs/` — forecasting metrics, predictions, feature importance, error summaries and prototype architecture.
- `redistribution_experiment_plan.md` — controlled pilot protocol.
- `tests/test_prototype.py` — 27 existing project tests.
- `tests/test_business_assistant_nlp.py` — 10 offline Business Assistant tests.
- `README.md`, `IMPLEMENTATION_README.md`, `DECISION_LOG.md` and `AI_USAGE.md` — setup, implementation, decisions and AI disclosure.

## 12. Conclusion

The project combines data quality auditing, exploratory analysis, non-parametric statistical testing, machine learning, scenario-based fleet planning and a Gemini-enabled Streamlit decision-support prototype. It identifies temporal and customer-segment differences in completed rentals and develops a Random Forest model with a latest recorded test R² of 0.8068. The live Business Assistant has demonstrated a successful Gemini function call, and the combined local test suite recorded 37 passing tests.

The strongest business contribution is the separation of what the data demonstrates from what still requires operational measurement. The recommended next move is to collect station-level evidence and run a controlled forecast-guided redistribution pilot before committing to broad fleet purchases.

---

**Primary dataset:** UCI Bike Sharing Dataset (`hour.csv`, `day.csv`), Capital Bikeshare, Washington, D.C., 2011–2012.  
**Technologies:** Python, Pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, Streamlit, Google Gemini API (`google-genai`), pytest and Jupyter Notebook.
