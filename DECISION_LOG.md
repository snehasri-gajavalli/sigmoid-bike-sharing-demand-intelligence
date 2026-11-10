**# Decision Log — Bike Sharing Demand Intelligence**



**\*\*Project:\*\*** Bike Sharing Demand Forecasting and Operational Decision Support  

**\*\*Case Study:\*\*** Sigmoid Data Science Trainee — Round 2  

**\*\*Dataset:\*\*** UCI Bike Sharing Dataset (2011–2012)



This document records the major analytical, statistical, modelling and business decisions made during the case study, along with the reasoning behind each decision.



**---**



**## 1. Business Problem Definition**



**\*\*Decision:\*\*** Investigate whether bike-sharing operational problems can be addressed through better demand forecasting and resource allocation before recommending additional fleet purchases.



**\*\*Reasoning:\*\*** The case describes riders experiencing bike unavailability while other bikes remain unused. This suggests a possible mismatch between the timing or location of demand and bike availability.



**\*\*Implication:\*\*** The analysis focuses on temporal demand patterns, customer behaviour, forecasting and illustrative fleet-planning scenarios.



**\*\*Limitation:\*\*** The available dataset contains completed rentals at an hourly citywide level. It does not directly record failed rental attempts, station-level availability or idle-bike locations. Therefore, redistribution is a recommendation to investigate and pilot, not a proven solution.



**## 2. Dataset Selection and Quality**



**\*\*Decision:\*\*** Use the UCI Bike Sharing Dataset, analysing both hourly and daily records.



**\*\*Reasoning:\*\*** The dataset provides historical rental counts, calendar variables, customer segments and weather conditions.



**\*\*Quality checks performed:\*\***

\- Checked dataset dimensions and column types.

\- Checked missing values and duplicate records.

\- Examined hourly timestamp continuity.

\- Investigated zero windspeed values.

\- Identified unusual rental-count observations.



**\*\*Decision on outliers:\*\*** Retain high-demand observations rather than automatically removing them.



**\*\*Reasoning:\*\*** High rental counts may represent genuine peak demand and are operationally important.



**## 3. Feature Engineering and Leakage Prevention**



**\*\*Decision:\*\*** Use calendar, weather and time-related variables as model inputs.



**\*\*Selected features:\*\***

\- Season

\- Year

\- Month

\- Hour

\- Holiday

\- Weekday

\- Working-day indicator

\- Weather situation

\- Temperature

\- Feels-like temperature

\- Humidity

\- Windspeed



**\*\*Target:\*\*** \`cnt\` (total hourly rentals).



**\*\*Excluded features:\*\***

\- \`casual\`

\- \`registered\`

\- \`instant\`



**\*\*Reasoning:\*\*** Casual and registered rentals sum to the target variable, so including them would introduce direct target leakage. The instant column is an identifier rather than a meaningful predictive feature.



**## 4. Exploratory Data Analysis**



**\*\*Decision:\*\*** Analyse demand separately by hour, working-day status, customer segment and weather condition.



**\*\*Reasoning:\*\*** Overall averages can hide important differences in demand behaviour.



**\*\*Key findings:\*\***

\- Average observed completed rentals per hour are approximately 189.5.

\- Maximum observed completed rentals in an hour are 977.

\- Working-day demand is concentrated around morning and evening commuting periods.

\- Non-working-day demand is relatively stronger around midday.

\- Registered users contribute a larger proportion of rentals on working days.

\- Casual users contribute a larger proportion on non-working days.

\- Adverse weather is associated with lower observed rental demand.



**\*\*Implication:\*\*** Operational planning should consider different demand profiles rather than using a uniform allocation strategy throughout the day.



**## 5. Statistical Testing**



**\*\*Decision:\*\*** Use non-parametric tests for comparisons where distributional assumptions may not be appropriate.



**\*\*Methods used:\*\***

\- Mann–Whitney U test for distributional comparisons.

\- Kruskal–Wallis test for comparisons across multiple groups.

\- Permutation testing for mean differences.

\- Bootstrap resampling for confidence intervals.



**\*\*Additional commute analysis decision:\*\*** Aggregate hourly records into daily observations and retain only complete days containing all 24 hours.



**\*\*Commute-hour definition:\*\*** 07:00–09:00 and 17:00–19:00.



**\*\*Observed result:\*\*** Mean commute-hour rental share was approximately 51.63% on working days and 26.25% on non-working days, based on 443 working days and 212 non-working days.



**\*\*Reasoning:\*\*** Daily aggregation avoids treating the 24 hourly records within the same day as independent observations.



**\*\*Limitation:\*\*** Consecutive days may still exhibit temporal dependence. Statistical significance does not establish causality.



**## 6. Model Development**



**\*\*Decision:\*\*** Compare a historical seasonal baseline with machine-learning regression models.



**\*\*Models evaluated:\*\***

\- Historical mean baseline grouped by working-day status and hour.

\- Random Forest Regressor.

\- Gradient Boosting Regressor.



**\*\*Reasoning:\*\*** A baseline establishes whether machine learning provides meaningful improvement over a simple historical estimate.



**\*\*Validation strategy:\*\*** Use a chronological split, training on earlier observations and evaluating on later observations.



**\*\*Reasoning:\*\*** This better represents forecasting future demand than a random train-test split.



**\*\*Leakage prevention:\*\*** Exclude customer segment counts and identifier columns from model features.



**## 7. Model Selection**



**\*\*Decision:\*\*** Retain the original Random Forest model as the primary model.



**\*\*Reasoning:\*\*** Random Forest was retained as the selected model in the current forecasting workflow. Model selection should be based on the chronological validation results; the final test set is used for the reported generalisation metrics.



**\*\*Original Random Forest configuration:\*\***

\- \`n_estimators = 300\`

\- \`max_depth = None\`

\- \`min_samples_split = 2\`

\- \`min_samples_leaf = 2\`

\- \`random_state = 42\`

\- \`n_jobs = -1\`



**\*\*Recorded test performance:\*\***

\- MAE: approximately 45.78

\- RMSE: approximately 71.02

\- R²: approximately 0.8955



**\*\*Tuning decision:\*\*** Retain the Random Forest configuration selected in the current forecasting workflow. Compare tuning alternatives using validation performance; do not use the final test set repeatedly to choose hyperparameters.



**\*\*Reasoning:\*\*** Hyperparameter tuning should be justified by actual validation improvements, not performed merely to produce a more complex model.



**## 8. Peak-Hour Error Analysis**



**\*\*Decision:\*\*** Evaluate model performance separately during commute periods and off-peak periods.



**\*\*Reasoning:\*\*** Overall performance metrics can hide errors during operationally important hours.



**\*\*Commute-hour windows:\*\*** 07:00–09:00 and 17:00–19:00.



**\*\*Recorded findings:\*\***

\- Morning peak MAE: approximately 64.21.

\- Evening peak MAE: approximately 99.11.

\- Off-peak MAE: approximately 33.79.

\- Working-day peak prediction bias: approximately -53.47.

\- Non-working-day peak prediction bias: approximately +12.68.



**\*\*Bias definition:\*\*** Prediction bias is calculated as predicted demand minus actual demand. Negative values indicate underprediction.



**\*\*Implication:\*\*** Forecasts require particular caution during working-day commute peaks. Operational planning should not rely solely on overall model accuracy.



**## 9. Fleet Estimation**



**\*\*Decision:\*\*** Use scenario-based fleet estimation rather than claiming an exact required fleet size.



**\*\*Formula:\*\***



\`Estimated fleet = (Hourly demand × Trip duration × Spatial buffer) / (1 − Maintenance share)\`



**\*\*Assumptions considered:\*\***

\- Average trip duration.

\- Spatial allocation buffer.

\- Maintenance reserve.



**\*\*Reasoning:\*\*** These assumptions allow sensitivity analysis when direct fleet inventory and trip-duration information are unavailable.



**\*\*Limitation:\*\*** Hourly rental starts are not equivalent to simultaneous bike occupancy. The estimates are illustrative scenarios, not verified fleet requirements.



**## 10. Business Recommendation**



**\*\*Decision:\*\*** Investigate demand-informed redistribution and operational improvements before committing to additional bike purchases.



**\*\*Reasoning:\*\***

\- Demand is concentrated in specific hours.

\- Working-day and non-working-day demand profiles differ.

\- The forecasting model provides useful demand estimates.

\- Existing data does not establish whether shortages are caused by insufficient total fleet capacity or poor spatial allocation.



**\*\*Recommended next step:\*\*** Conduct an operational pilot using demand forecasts to guide redistribution, then measure its impact against the existing allocation process.



**\*\*Additional data required:\*\***

\- Station-level bike availability.

\- Failed or abandoned rental requests.

\- Trip origins and destinations.

\- Trip duration.

\- Bike maintenance status.

\- Rebalancing and relocation records.



**## 11. Dashboard and Business Assistant**



**\*\*Decision:\*\*** Build an interactive Streamlit dashboard to communicate the analytical findings and demonstrate practical decision-support capabilities.



**\*\*Dashboard sections:\*\***

1\. Demand Overview

2\. Hourly Analysis

3\. Customer Segmentation

4\. Demand Prediction

5\. Fleet Planning & Experiment

6\. Business Assistant



**\*\*Business Assistant decision:\*\*** Use a deterministic, rule-based approach to answer predefined business questions using the project's analytical results.



**\*\*Reasoning:\*\*** This provides a lightweight, transparent prototype without requiring an external language-model API.



**\*\*Limitation:\*\*** The assistant is not a general-purpose conversational AI system.



**\*\*Automated testing status:\*\*** The latest local test run completed with 27 passed, 0 failed and 0 skipped. One third-party deprecation warning was displayed and did not affect the result.



**---**



**## Final Decision**



The case study does not provide sufficient evidence to conclude that the company must purchase additional bikes or that redistribution alone will solve the problem.



The most defensible recommendation is to investigate demand-informed redistribution and fleet-health improvements through a controlled operational pilot, collect station-level availability and unmet-demand data, and use those results to make a better-informed fleet expansion decision.
