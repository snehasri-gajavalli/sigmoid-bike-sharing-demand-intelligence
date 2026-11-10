# Bike Sharing Demand Intelligence
## Final Case Study Report — Sigmoid Data Science Trainee, Round 2

**Dataset:** UCI Bike Sharing Dataset — Capital Bikeshare, Washington, D.C. (2011–2012)  
**Domain:** Demand analysis, forecasting, fleet planning and natural-language decision support  
**Prototype:** Gemini-enabled Streamlit Business Assistant

## Executive summary

This case study investigates whether a bike-sharing operator should buy more bikes or first improve allocation and operations when riders report bike unavailability while bikes appear unused elsewhere.

The analysis uses 17,379 hourly records and 731 daily records. It identifies differences by hour, day type and customer segment, and associations with weather. A Random Forest Regressor predicts hourly **completed rentals** with a latest chronological test MAE of 64.09, RMSE of 94.54 and R² of 0.8068. A Little's Law scenario gives 491 bikes under explicit assumptions; this is not a measured fleet requirement. A six-area Streamlit dashboard includes a Gemini-powered natural-language Business Assistant with validated manual function calling.

**Recommendation:** Do not recommend broad fleet expansion from citywide completed-rental counts alone. First collect station availability, failed rental attempts and fleet-health data, then test forecast-guided redistribution in a controlled pilot. Consider targeted purchases only if verified shortages persist after redistribution and repairs.

## 1. Business problem and scope

The reported situation could result from temporal or spatial imbalance, poorly timed redistribution, unavailable or damaged bikes, technology failures or insufficient capacity. The supplied dataset can analyse temporal demand patterns but cannot identify the actual cause.

Keep these concepts separate:
- **Observed completed rentals:** Recorded in `hour.csv` and `day.csv`.
- **Predicted completed rentals:** Estimated by the trained model.
- **Unobserved/suppressed demand:** Not recorded in the supplied dataset.
- **Scenario fleet estimates:** Calculations based on assumptions.
- **Actual fleet sufficiency and station availability:** Cannot be measured without station-level operational data.

This distinction prevents predictions or scenario estimates from being presented as proof of stock-outs.

## 2. Dataset, quality and preparation

### Dataset

| File | Rows | Description |
|---|---:|---|
| `hour.csv` | 17,379 | Hourly completed-rental records |
| `day.csv` | 731 | Daily completed-rental records |

The target is `cnt` (total completed rentals). Candidate predictors include season, year, month, weekday, holiday, working-day status, hour, weather situation, temperature, feels-like temperature, humidity and windspeed. `casual` and `registered` describe customer segments.

### Quality checks

- Recorded-row null, duplicate-row and duplicate date-hour key checks were reported as zero.
- Timestamp continuity checks found **165 missing hourly timestamps across 76 incomplete days**. These are absent timestamps, not nulls in existing rows.
- Missing timestamps were not automatically converted to zero rentals.
- Checked `casual + registered = cnt`.
- Retained high-rental observations unless independent evidence showed a data error.
- Flagged zero humidity and windspeed readings for interpretation rather than automatically replacing them.
- Weather category 4 has only three observations; conclusions about it are unreliable.

Filling absent timestamps with zero could distort demand profiles. Removing genuine peaks could also weaken the model's ability to learn high-demand periods.

### Feature engineering and leakage prevention

Calendar/time and weather variables were used to predict `cnt`. `casual` and `registered` were excluded because they sum directly to the target and would cause **target leakage**. `instant` was excluded because it is only a row identifier.

For live forecasting, future weather must come from a forecast available at decision time, not future observed weather.

## 3. Exploratory data analysis

### Overall and hourly demand

- Total recorded hourly rentals: **3,292,679**
- Average: **189.5 completed rentals per recorded hour**
- Maximum observed hourly count: **977**
- Highest average hour: **17:00 — 461.45 rentals/hour**
- Next highest: **18:00 — 425.51 rentals/hour**

These are citywide timing results; they do not show which individual stations need bikes.

### Working and non-working days

| Day type | Average rentals/hour |
|---|---:|
| Working day | 193.21 |
| Non-working day | 181.41 |

Working-day average demand is approximately **6.51% higher**. Working days show stronger morning and evening commute peaks (07:00–09:00 and 17:00–19:00); non-working-day demand is relatively more concentrated around midday (12:00–15:00).

### Customer segmentation

| Day type | Casual share | Registered share |
|---|---:|---:|
| Working day | 13.23% | 86.77% |
| Non-working day | 31.66% | 68.34% |

Registered customers contribute a larger share on working days; casual customers contribute a larger share on non-working days. Day-type-specific profiles may therefore help operational planning.

### Weather associations

| Weather category | Average hourly rentals |
|---|---:|
| Clear | 204.87 |
| Mist/cloudy | 175.17 |
| Light rain/snow | 111.58 |
| Heavy rain/snow | 74.33 |

Less favourable weather categories are associated with lower average completed rentals. This is observational and does not prove causation. Heavy rain/snow has only three observations and should not be over-interpreted.

## 4. Statistical analysis

### Primary hypothesis: commute-window concentration

- **H0:** The share of daily rentals in commute windows is the same on working and non-working days.
- **H1:** The share is higher on working days.
- **Windows:** 07:00–09:00 and 17:00–19:00 (hours 7, 8, 9, 17, 18 and 19).
- **Unit of analysis:** One complete day, not one hourly row. Days without all 24 hourly observations were excluded.

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
| Bootstrap 95% CI | 24.67–26.06 percentage points |
| Cliff's delta | 0.9897 |

The results support a very large historical difference in commute-window concentration. Consecutive days may be temporally dependent, which can affect inferential uncertainty. The result does not prove station-level shortages or origin-destination flows.

Other comparisons examined AM, PM and midday periods, weather groups and seasonal groups using non-parametric comparisons and adjusted pairwise tests. The selected high-temperature hypothesis was not statistically significant at the chosen threshold; this does not prove temperature has no relationship with demand.

## 5. Hourly demand forecasting

### Target and predictors

The model predicts hourly **completed rentals** (`cnt`) using calendar, time and weather features. Customer segment counts are excluded because they sum to the target.

### Chronological split

A chronological 70% / 15% / 15% train-validation-test split preserves time order.

| Partition | Rows | Date range |
|---|---:|---|
| Training | 12,165 | 2011-01-01 to 2012-05-27 |
| Validation | 2,607 | 2012-05-27 to 2012-09-12 |
| Test | 2,607 | 2012-09-12 to 2012-12-31 |

Earlier observations are used for fitting, the next period for model selection and the latest period for final testing. The test period is not used to choose the model.

### Final model performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest Regressor | 64.09 | 94.54 | 0.8068 |

- **MAE:** Average absolute prediction error, in rentals/hour.
- **RMSE:** Gives greater weight to larger prediction errors.
- **R²:** Describes the proportion of target variation explained on the test observations.

These are the latest recorded results and supersede earlier model-run metrics.

### Feature importance

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

`atemp` is the strongest individual encoded feature in the recorded output. One-hot encoded hour columns should not be interpreted as the total importance of the hour variable. Feature importance indicates model reliance, not causation.

### Error analysis

Held-out test predictions were compared across morning peak, evening peak and off-peak periods. Peak-period errors were higher than off-peak errors, with the evening peak the most difficult period. Bias is defined as **predicted demand minus actual demand**; negative bias means underprediction.

The model predicts completed rentals, not latent demand, unserved requests or station-specific stock-outs. This matters because underprediction during peak periods could lead to insufficient preparation if forecasts are used operationally.

## 6. Fleet planning and scenario analysis

### Little's Law

Little's Law estimates the average number of bikes in use:

`L = λ × W`

where `L` is average bikes in motion, `λ` is hourly rental starts and `W` is average trip duration in hours. A spatial positioning buffer and maintenance reserve are additional assumptions.

### Illustrative scenario

| Input/result | Value | Status |
|---|---:|---|
| 2012 P95 demand reference | 662 rentals/hour | Historical reference |
| Average trip duration | 20 minutes | Assumption |
| Spatial positioning buffer | 2.0× | Assumption |
| Maintenance reserve | 10% | Assumption |
| Bikes in motion | 220.7 | Calculated |
| Positioned bikes before reserve | 441.3 | Calculated |
| Scenario fleet after reserve | **491 bikes** | Calculated, rounded up |

Calculation:

`ceil(662 × (20/60) × 2.0 ÷ (1 − 0.10)) = 491 bikes`

This is **not a measured fleet requirement**. Actual fleet sufficiency depends on station inventory, trip duration, returns, docks, maintenance and spatial distribution.

### Sensitivity

At 662 rentals/hour and 10% maintenance:

| Trip duration | 1.5× buffer | 2.0× buffer | 3.0× buffer |
|---|---:|---:|---:|
| 15 minutes | 276 | 368 | 552 |
| 20 minutes | 368 | 491 | 736 |
| 30 minutes | 552 | 736 | 1,104 |

The optional station-transfer shortlist is a heuristic, not a route optimizer. Synthetic test data validates the software path, not real operating savings.

## 7. Streamlit dashboard and Gemini Business Assistant

### Dashboard

The Streamlit application has six areas:
1. **Demand Overview** — rental statistics and patterns.
2. **Hourly Analysis** — demand by hour and day type.
3. **Customer Segmentation** — casual and registered comparisons.
4. **Demand Prediction** — hourly completed-rental estimates.
5. **Fleet Planning & Experiment** — percentiles, Little's Law scenarios, sensitivity and optional station-transfer heuristic.
6. **Business Assistant** — natural-language questions, evidence-backed answers and data limitations.

### Gemini function-calling workflow

The Gemini-enabled entry point is `app_GEMINI_NLP.py`; the original `app.py` is retained.

1. A manager enters a plain-English question.
2. Gemini selects from an allow-list of analytical tools.
3. The local dispatcher validates the function name and arguments.
4. Approved Python functions run against local Pandas data or the saved model.
5. Aggregate results are returned to Gemini for explanation.
6. A numeric-grounding check checks generated claims against tool results.
7. If the API key is missing or Gemini fails, a labelled deterministic rule-based fallback is used.

Gemini does not execute arbitrary Python. The API key is stored in environment variables or Streamlit secrets, not in source code.

**Launch command:** `streamlit run app_GEMINI_NLP.py`

**Prototype verification:** A live Gemini response to “When is demand highest?” selected `get_hourly_demand_profile`, identified 17:00 at 461.45 completed rentals/hour and passed the grounding check. Day-type and customer-segment questions were also manually tried. Automated assistant tests are offline and do not call the live Gemini API.

## 8. Automated testing and deliverables

**Test command:**
```powershell
python -m pytest tests -v
```

**Recorded result (10 November 2026):** 37 passed, 0 failed, 1 warning (27 project tests and 10 Business Assistant tests). The warning did not cause a test failure.

Key deliverables:
- Four Jupyter notebooks: data understanding, EDA, hypothesis testing and demand forecasting.
- `app_GEMINI_NLP.py` and retained `app.py`.
- `business_assistant_nlp.py` — function schemas, allow-listed dispatcher, validation, grounding and fallback.
- `fleet_planning.py` — fleet scenarios, percentiles, sensitivity and demand-timing functions.
- `models/bike_demand_model_bundle.pkl` — current model and preprocessing bundle (Git LFS).
- `outputs/` — metrics, predictions, feature importance, error summaries and architecture.
- `redistribution_experiment_plan.md`, tests, README and implementation documentation.

Repository safety: `.gitignore` excludes `.env`, Streamlit secrets, virtual environments, caches and local editor settings. The legacy model is excluded; the current model bundle is tracked with Git LFS because it exceeds GitHub's regular 100 MiB file limit.

## 9. Recommendation, pilot and limitations

### Recommendation

1. Instrument station inventory, failed rental attempts, trip records and bike health.
2. Use day-type-specific demand profiles to support station-level forecasting and operational planning.
3. Run a 4–6 week matched-cluster forecast-guided redistribution pilot against business-as-usual.
4. Pre-register the primary outcome, peak windows, exclusions, guardrails and decision threshold.
5. Consider targeted purchases only if verified unmet demand persists after redistribution and fleet-health improvements.

**Primary pilot metric:** Empty-station minutes per station-hour during pre-defined peak windows.  
**Secondary metrics:** Failed rental attempts per 1,000 attempts, completed rentals per station-hour, target-stock availability, idle-bike hours and complaints.  
**Guardrails:** Driver hours and distance, costs, cost per additional successful rental, safety incidents, bike damage and control-zone service quality.

### Limitations and required data

The supplied dataset has no:
- Station-level rentable-bike or empty-dock inventory.
- Failed or abandoned rental attempts and reason codes.
- Trip-level start/end stations or actual trip duration.
- Bike health or maintenance downtime.
- Rebalancing movements, driver hours, distance or cost records.

The model predicts completed rentals, not suppressed demand or actual shortages. Fleet scenarios depend on assumed trip duration, positioning buffer and maintenance reserve. Weather category 4 has only three observations, and temporal dependence may affect statistical uncertainty.

Minimum additional data: timestamped station inventory snapshots, trip start/end records, failed-rental logs, bike health and rebalancing movement/cost records.

## Conclusion

The project combines data-quality auditing, exploratory analysis, non-parametric statistical testing, machine learning, scenario-based fleet planning and a Gemini-enabled Streamlit decision-support prototype. It identifies temporal and customer-segment differences in completed rentals and produces a Random Forest model with a latest recorded test R² of 0.8068. The live Business Assistant demonstrated a successful Gemini function call, and the combined local test suite recorded 37 passing tests.

The key contribution is separating what the historical data demonstrates from what still needs operational measurement. The recommended next step is to collect station-level evidence and run a controlled forecast-guided redistribution pilot before committing to broad fleet purchases.

**Technologies:** Python, Pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, Streamlit, Google Gemini API (`google-genai`), pytest and Jupyter Notebook.
