**# Bike Sharing Demand Analysis and Forecasting**



**### Data Science Case Study – Sigmoid**



**\*\*Project Domain:\*\*** Data Science | Demand Forecasting | Business Analytics



**---**



**# 1. Executive Summary**



This project analyzes bike-sharing rental demand to identify usage patterns, understand the factors influencing demand, and develop a machine learning model for hourly demand forecasting.



Using the UCI Bike Sharing Dataset, the analysis examines hourly and daily rental patterns, customer segmentation, weather conditions, working-day effects, and demand variations.



A Random Forest Regressor was developed for hourly demand prediction. On the latest recorded chronological test run, it achieved an MAE of 64.09 rentals, an RMSE of 94.54 rentals, and an R² of 0.8068. These are the metrics recorded in the project's current `forecasting_test_metrics.csv` output.



The findings indicate that demand varies considerably by hour and day type. Working days exhibit strong morning and evening peaks, while non-working days show stronger midday demand.



The project concludes with operational recommendations focused on demand-informed redistribution and further investigation before making additional fleet investments.



**# 2. Business Problem**



Bike-sharing operators may encounter situations where customers cannot find available bikes in certain locations while bikes remain underutilized elsewhere.



This creates several business challenges:



\- Lost rental opportunities.

\- Uneven utilization of available bikes.

\- Inefficient fleet allocation.

\- Increased operational costs.

\- Potentially unnecessary fleet expansion.



The central business question is:



**\*\*Can historical demand patterns and machine learning help improve bike allocation decisions before investing in additional bikes?\*\***



**## 2.1 Project Objectives**



1\. Understand historical bike rental demand.

2\. Identify peak-demand hours and days.

3\. Analyze the influence of weather and working days.

4\. Understand differences between casual and registered customers.

5\. Develop an hourly demand forecasting model.

6\. Evaluate model performance and prediction errors.

7\. Translate analytical findings into operational recommendations.



**# 3. Dataset Description**



The project uses the UCI Bike Sharing Dataset.



Two datasets were used:



\| Dataset | Records | Description |

\|---|---:|---|

\| hour.csv | 17,379 | Hourly rental records |

\| day.csv | 731 | Daily rental records |



The target variable is \`cnt\`, representing the total number of bike rentals.



Important variables include:



\| Feature | Description |

\|---|---|

\| season | Season category |

\| yr | Year |

\| mnth | Month |

\| hr | Hour of the day |

\| holiday | Holiday indicator |

\| weekday | Day of the week |

\| workingday | Working-day indicator |

\| weathersit | Weather category |

\| temp | Normalized temperature |

\| atemp | Normalized feels-like temperature |

\| hum | Normalized humidity |

\| windspeed | Normalized windspeed |

\| casual | Casual rentals |

\| registered | Registered rentals |

\| cnt | Total rentals |



**# 4. Data Understanding and Preparation**



The initial data exploration covered:



\- Dataset dimensions and data types.

\- Missing-value checks.

\- Duplicate checks.

\- Target-variable consistency.

\- Timestamp continuity.

\- Outlier investigation.



**## 4.1 Data Quality Findings**



The hourly dataset contains 17,379 recorded observations.



No missing values were found in the recorded rows. However, timestamp analysis identified 165 missing hourly timestamps across 76 incomplete days.



This distinction is important: missing timestamps represent absent observations, not necessarily missing values inside existing records.



The relationship between customer segments and total rentals was also validated:



\`casual + registered = cnt\`



**## 4.2 Outlier Treatment**



High-demand observations were retained.



These observations may represent genuine peak-demand periods and are important for operational planning. Removing them could reduce the model's ability to learn demand spikes.



**# 5. Exploratory Data Analysis**



**## 5.1 Overall Demand**



The dataset contains approximately 3.29 million total rentals.



The average hourly demand is approximately 189.5 rentals, while the maximum observed hourly demand is 977 rentals.



This indicates substantial variation in demand across different hours and conditions.



**## 5.2 Hourly Demand Patterns**



The overall demand peak occurs around 17:00, followed by 18:00.



Average demand:



\| Hour | Average rentals |

\|---|---:|

\| 17:00 | 461.45 |

\| 18:00 | 425.51 |



These patterns suggest that evening periods are particularly important for operational planning.



**## 5.3 Working-Day vs Non-Working-Day Analysis**



Working days exhibit strong commute-related demand peaks.



\| Hour | Working-day demand | Non-working-day demand |

\|---|---|---|

\| 07:00 | Higher | Lower |

\| 08:00 | Higher | Lower |

\| 17:00 | Higher | Lower |

\| 18:00 | Higher | Lower |

\| 12:00–15:00 | Relatively lower | Relatively higher |



The average hourly demand was approximately:



\- Working days: 193.21 rentals.

\- Non-working days: 181.41 rentals.



Working-day demand was approximately 6.51% higher overall.



The largest differences were observed during morning and evening commute hours.



**## 5.4 Customer Segmentation**



The analysis compared casual and registered customers.



\| Day type | Casual share | Registered share |

\|---|---:|---:|

\| Working day | 13.23% | 86.77% |

\| Non-working day | 31.66% | 68.34% |



Registered customers account for a significantly larger proportion of working-day rentals.



Casual customers contribute a larger proportion of non-working-day rentals.



This suggests that customer behavior differs depending on the type of day.



**## 5.5 Weather Analysis**



Average hourly demand across weather categories:



\| Weather category | Average rentals |

\|---|---:|

\| Clear | 204.87 |

\| Mist / Cloudy | 175.17 |

\| Light rain / snow | 111.58 |

\| Heavy rain / snow | 74.33 |



Demand decreases as weather conditions become less favorable.



However, the heavy-rain category contains only three observations. Therefore, its average should not be treated as a reliable general estimate.



**# 6. Statistical Analysis and Hypothesis Testing**



Statistical testing was used to examine whether observed demand differences were supported by the data.



The analysis included:



\- Mann-Whitney U tests.

\- Kruskal-Wallis tests.

\- Pairwise comparisons.

\- Working-day and non-working-day comparisons.

\- Weather-related comparisons.



**## 6.1 Working-Day Demand**



The analysis found a meaningful difference in hourly demand patterns between working days and non-working days.



In particular, working days showed stronger demand during morning and evening commute periods.



**## 6.2 Temperature Hypothesis**



The temperature-related hypothesis was not statistically significant at the selected high-temperature thresholds.



Therefore, the analysis did not establish sufficient statistical evidence for the proposed difference under those specific thresholds.



This does not prove that temperature has no relationship with bike demand.



**## 6.3 Daily-Level Commute Demand Analysis**



To investigate whether commute-hour demand differs between working and non-working days, hourly rentals were aggregated at the daily level.



Commute hours were defined as:



\- Morning: 07:00–09:00.

\- Evening: 17:00–19:00.



Only complete days containing all 24 hourly observations were included in this analysis.



\| Metric | Working Days | Non-Working Days |

\|---|---:|---:|

\| Number of days | 443 | 212 |

\| Mean commute-hour share | 51.63% | 26.25% |

\| Median commute-hour share | 52.38% | 26.31% |



The difference in mean commute-hour share was approximately 25.39 percentage points.



Statistical testing produced the following results:



\| Test | Result |

\|---|---:|

\| Mann-Whitney U statistic | 93,434 |

\| Mann-Whitney p-value | 1.71 × 10⁻⁹³ |

\| Permutation test p-value | 0.0002 |

\| Bootstrap 95% confidence interval | 24.67–26.06 percentage points |

\| Cliff's delta | 0.9897 |



These results indicate a substantial difference in commute-hour concentration between working and non-working days.



The analysis supports the importance of separate operational planning for different day types.



However, the observations are consecutive days. Temporal dependence may affect the uncertainty estimates, and the statistical results should be interpreted with this limitation in mind.



**# 7. Machine Learning: Hourly Demand Forecasting**



The next objective was to predict hourly bike rental demand using historical data.



**## 7.1 Target Variable**



\`cnt\` – Total hourly bike rentals.



**## 7.2 Input Features**



The model used:



\- Season

\- Year

\- Month

\- Hour

\- Holiday

\- Weekday

\- Working day

\- Weather situation

\- Temperature

\- Feels-like temperature

\- Humidity

\- Windspeed



**## 7.3 Train-Test Split**



A chronological 70/15/15 split was used instead of random shuffling to preserve temporal ordering. The earliest 70% of observations were used for training, the next 15% for validation and model selection, and the latest 15% for final testing.



For 17,379 modelling observations, this gives 12,165 training rows, 2,607 validation rows and 2,607 test rows. The exact date boundaries are printed by the forecasting notebook when it runs.



This approach evaluates the model on later observations that were not used for fitting or validation.



**# 8. Model Development and Comparison**



The forecasting notebook evaluates a historical hour-and-working-day mean baseline and a Random Forest Regressor. Model selection is performed using the chronological validation period; the final test period is reserved for reporting generalisation performance.



**## 8.1 Latest Recorded Test Performance**



\| Model | MAE | RMSE | R² |

\|---|---:|---:|---:|

\| Random Forest | 64.09 | 94.54 | 0.8068 |



These values are taken from the current `forecasting_test_metrics.csv`. Do not mix them with metrics from earlier model runs or different train-test splits. The baseline's latest test metrics should be added to this table only after they are regenerated from the same final test split.



**## 8.2 Model Selection**



The current forecasting output identifies Random Forest as the selected model. Its latest recorded test performance is:



\- **\*\*MAE:\*\*** 64.09 rentals.

\- **\*\*RMSE:\*\*** 94.54 rentals.

\- **\*\*R²:\*\*** 0.8068.



The selected fitted model and its preprocessing feature schema are saved together using Joblib for use in the Streamlit application. These metrics describe the recorded test run; they should be regenerated whenever the model or split changes.



**# 9. Feature Importance**



Feature importance analysis was performed to understand which variables contributed most to the Random Forest predictions.



The latest feature-importance output ranks the following encoded features highest:



\| Encoded feature | Importance |

\|---|---:|

\| atemp | 14.83% |

\| hr_17 | 9.38% |

\| hum | 8.54% |

\| hr_18 | 8.14% |

\| temp | 6.02% |

\| hr_8 | 5.90% |

\| workingday_1 | 4.38% |

\| workingday_0 | 4.15% |

\| hr_19 | 2.90% |



The strongest individual feature is feels-like temperature (`atemp`), followed by the 17:00 and 18:00 hour indicators. Because categorical variables are one-hot encoded, separate hour indicators appear as separate features; their individual importances should not be described as the total importance of the hour variable.



Feature importance indicates predictive contribution within the trained model; it does not establish causation.



**# 10. Model Error Analysis**



The model's prediction errors were examined across different operating conditions.



Important observations:



\- Underprediction was observed during working-day evening commute periods.

\- Errors were particularly noticeable around 17:00 and 18:00.

\- Weather category 3 showed higher prediction errors.

\- Weather category 4 had very few observations, making its error estimates less reliable.



This analysis indicates that overall model performance should not be the only evaluation criterion. Performance during operationally important peak periods also matters.



**## 10.1 Peak-Hour Prediction Performance**



The model was evaluated separately across morning peak, evening peak and off-peak periods.



\| Period | MAE | RMSE | Prediction Bias |

\|---|---:|---:|---:|

\| Morning peak | 64.21 | 88.11 | -21.11 |

\| Evening peak | 99.11 | 126.14 | -43.00 |

\| Off-peak | 33.79 | 52.65 | -13.34 |



Prediction bias is calculated as:



\`Prediction Bias = Predicted Demand - Actual Demand\`



A negative bias indicates underprediction.



The model shows its largest prediction errors during evening peak hours. The negative bias indicates systematic underprediction during these periods.



This finding is operationally important because underestimating evening demand could lead to insufficient bike availability if forecasts are used for planning.



Therefore, peak-period performance should be considered alongside overall model accuracy.



**# 11. Fleet Planning and Sensitivity Analysis**



Fleet-sizing scenarios were explored using historical and predicted demand distributions.



Important demand reference points:



\| Measure | Rentals per hour |

\|---|---:|

\| Historical 95th percentile | 702.25 |

\| Historical maximum | 977 |

\| Predicted 95th percentile | 622.76 |



These values provide demand reference points for scenario analysis.



An illustrative fleet calculation can be expressed as:



\`Estimated Fleet = ceil(Hourly Demand × Trip Duration × Spatial Buffer / (1 - Maintenance Share))\`



For example, using:



\- Hourly demand = 702.25

\- Trip duration = 0.75 hours

\- Spatial buffer = 1.2

\- Maintenance share = 10%



The illustrative estimated fleet requirement is 703 bikes.



However, hourly rental counts cannot directly determine the number of bikes required.



Actual fleet requirements depend on factors such as:



\- Trip duration.

\- Bike availability.

\- Station-level demand.

\- Bike arrival and departure flows.

\- Maintenance and downtime.

\- Unfulfilled customer requests.



Hourly rental starts do not necessarily represent simultaneous bike occupancy.



Therefore, fleet estimates should be treated as illustrative scenarios based on assumptions, not as confirmed fleet requirements.



**# 12. Business Recommendations**



Based on the analysis, the following recommendations are proposed.



**## 12.1 Demand-Informed Redistribution**



Investigate redistribution strategies around predictable demand peaks.



Working-day morning and evening commute periods deserve particular attention.



**## 12.2 Different Strategies for Different Day Types**



Use different operational plans for working days and non-working days.



Working days show stronger commute-related peaks, while non-working days have relatively stronger midday demand.



**## 12.3 Weather-Aware Planning**



Consider weather conditions when planning operational capacity.



Demand tends to be lower under adverse weather conditions, although the heavy-rain observations are too limited for strong conclusions.



**## 12.4 Validate Before Expanding the Fleet**



Do not recommend purchasing additional bikes solely from citywide rental counts.



First investigate station-level availability, idle-bike distribution and unmet customer demand.



A controlled redistribution pilot can help determine whether operational changes improve availability and utilization.



**# 13. Interactive Streamlit Dashboard**



An interactive Streamlit dashboard was developed to make the analysis accessible to business users.



The application includes six sections:



1\. **\*\*Demand Overview:\*\*** Displays overall rental statistics and demand patterns.

2\. **\*\*Hourly Analysis:\*\*** Examines demand variations across different hours and day types.

3\. **\*\*Customer Segmentation:\*\*** Compares casual and registered users.

4\. **\*\*Demand Prediction:\*\*** Uses the saved forecasting model to generate demand estimates.

5\. **\*\*Fleet Planning & Experiment:\*\*** Explores fleet-sizing scenarios, sensitivity assumptions and an optional station-transfer shortlist.

6\. **\*\*Business Assistant:\*\*** Provides rule-based business insights based on predefined questions.



The dashboard connects the analytical results with practical business interpretation.



**# 14. Limitations**



The project uses citywide hourly rental data rather than station-level operational data.



Consequently, the dataset does not directly identify:



\- Stations with no available bikes.

\- Stations with excess idle bikes.

\- Failed rental attempts.

\- Actual trip durations.

\- Station-level arrival and departure flows.

\- Maintenance-related bike unavailability.



Therefore, this analysis identifies demand patterns but cannot independently prove that station-level bike shortages or redistribution inefficiencies occurred.



The model also relies on historical patterns and may not capture unexpected events or sudden changes in demand.



**# 15. Future Scope**



The project can be extended through:



1\. Integration of real-time station availability data.

2\. Station-level demand forecasting.

3\. Incorporation of trip duration and destination information.

4\. Identification of potential bike shortage and surplus locations.

5\. Optimization of redistribution routes.

6\. Evaluation of redistribution strategies through controlled pilots.

7\. Development of an LLM-powered business assistant.



These improvements would help transform citywide demand forecasting into a more comprehensive bike-sharing fleet optimization system.



**# 16. Automated Testing**



An automated test suite was developed using Pytest to validate the project.



The suite contains 27 tests covering dataset integrity, target consistency, model-artifact availability and loading, model prediction interfaces, fleet calculations and invalid inputs, observed demand percentiles, fleet sensitivity, demand-timing recommendations, station-transfer planning and customer-segment validation.



The tests were executed using:



\`\`\`bash

python -m pytest tests/test_prototype.py -v

\`\`\`



**\*\*Latest test result:\*\*** 27 passed, 0 failed and 0 skipped in the tested local environment. One third-party `anyio` deprecation warning was reported; it did not affect the test results.



These tests provide basic validation of the model, dataset and fleet calculation logic.



**# 17. Conclusion**



This project demonstrates how exploratory data analysis, statistical testing and machine learning can be combined to understand bike-sharing demand and support operational decision-making.



The latest recorded Random Forest test run achieved an R² of 0.8068, with an MAE of 64.09 rentals and an RMSE of 94.54 rentals. This indicates useful predictive signal, while the peak-period error analysis highlights opportunities for further improvement.



The analysis also identified important differences between working-day and non-working-day demand, customer segments and weather conditions.



The peak-hour error analysis further highlighted that evening demand is more difficult for the model to predict accurately.



Most importantly, the findings suggest that operators should investigate demand-informed redistribution and station-level availability before committing to additional fleet purchases.



The project provides a foundation for more advanced demand forecasting and operational optimization using real-time station-level data.



**---**



**\*\*Technologies:\*\*** Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | SciPy | Streamlit | Pytest | Jupyter Notebook



**\*\*Dataset:\*\*** UCI Bike Sharing Dataset



**\*\*Project:\*\*** Sigmoid Data Science Case Study