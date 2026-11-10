# AI Usage Disclosure

**Project:** Bike Sharing Demand Intelligence  
**Case study:** Sigmoid Data Science Trainee — Round 2  
**Approach:** Human-led, AI-assisted development and validation

## 1. Purpose and scope

AI tools were used as supporting resources during project development. They assisted with learning unfamiliar concepts, planning analyses, drafting and refining code, debugging, interpreting outputs, preparing visualizations, building the Streamlit interface, writing tests and preparing documentation. AI assistance did not remove the project owner's responsibility to understand, run, check and explain the work.

## 2. Tools used

- **ChatGPT (OpenAI):** used for explanations, analysis planning, code drafts and refinements, debugging support, interpretation of project outputs, report/README/decision-log drafting, UI guidance and test-authoring assistance.
- **Google Gemini API:** integrated into the Business Assistant as the natural-language layer. Gemini interprets a manager's question and selects from a defined set of analytical functions. The local Python functions—not Gemini—perform the calculations and access the Pandas data/model. Gemini receives aggregate tool results for explanation rather than the raw CSV files or arbitrary executable code.
- **Python libraries and project code:** Pandas, NumPy, scikit-learn, SciPy, Matplotlib/Seaborn, Streamlit, joblib and pytest were used in the data workflow, modelling, application and validation.

## 3. Areas where AI assistance was used

### Data understanding and exploratory analysis

AI assistance helped explain dataset fields, timestamp gaps, data-quality checks, feature meanings and exploratory-analysis approaches. It helped organize questions about hourly demand, day type, customer segment and weather. Findings in the report are based on the project's dataset and recorded outputs, not on unverified model-generated numbers.

### Statistical analysis

AI assistance helped explain null/alternative hypotheses, daily-level aggregation, non-parametric tests, p-values, confidence intervals, effect sizes and interpretation limits. The commute-window hypothesis uses complete days as the analysis unit to avoid treating the 24 hourly rows within a day as independent observations.

### Machine learning

AI assistance helped structure the Random Forest workflow, chronological train/validation/test evaluation, feature-importance interpretation and peak-period error analysis. The reported final model metrics are the latest recorded project test values: MAE 64.09, RMSE 94.54 and R² 0.8068. Earlier model-run metrics were not mixed into the final result table.

### Fleet planning and business reasoning

AI assistance helped formulate the Little's Law scenario, identify trip-duration/positioning/maintenance assumptions, structure sensitivity analysis and develop the controlled redistribution pilot proposal. Fleet scenario numbers are explicitly labelled assumption-based. AI assistance also helped distinguish what citywide completed-rental data can show from what requires station-level operational records.

### Streamlit and Gemini Business Assistant

AI assistance was used to draft and debug the Gemini integration, function declarations, argument validation, local dispatcher, fallback behaviour and UI integration. The implemented design uses manual function calling and an allow-listed set of analytical functions. A numeric-grounding check is applied to the generated explanation. The original `app.py` was retained and the Gemini-enabled entry point is `app_GEMINI_NLP.py`.

### Documentation and visual communication

AI assistance helped organize and edit the README, decision log, AI-use disclosure, case-study report, architecture diagram and explanatory chart captions. The project owner should retain the notebooks, source files and output CSVs as the evidence trail for reported values.

## 4. Human review and responsibility

The project owner defined the case-study objective, ran the local commands and notebooks, inspected the Streamlit dashboard, configured the Gemini API key locally, tested Business Assistant questions, ran the pytest suites and made the final project recommendation. AI-generated code and interpretations were treated as drafts or assistance and were not assumed correct without execution or review.

The owner remains responsible for understanding and defending the analysis, model assumptions, experiment design, code, test results and final recommendation during evaluation or interview.

## 5. Validation evidence

- Existing project test suite: 27 tests passed.
- Business Assistant offline test suite: 10 tests passed.
- Combined run: `python -m pytest tests -v` — **37 passed, 0 failed, 1 warning** (local run recorded on 10 November 2026).
- A live Gemini function-calling response was separately observed in the local Streamlit application. The offline pytest suite does not call the live Gemini API.

## 6. Responsible use and limitations

- AI-generated code can be incorrect and must be run and tested.
- Observational relationships do not establish causation.
- The Random Forest predicts completed rentals, not unobserved demand or station-level shortages.
- The 491-bike example is a scenario using assumed trip duration, spatial buffer and maintenance reserve, not a verified fleet requirement.
- The UCI dataset lacks station availability, failed rental attempts, trip durations, station-to-station flows and maintenance status.
- Weather category 4 has only three observations, and consecutive daily data may exhibit temporal dependence.
- The Gemini API key is supplied through a local environment variable or Streamlit secrets. It must not be committed to source control or exposed in screenshots.

## 7. Disclosure statement

This case study was developed with AI assistance for learning, coding, debugging, statistical interpretation, prototype development and documentation. The project owner executed the work in the local project environment, reviewed outputs, tested the application and retained responsibility for the final analytical and business decisions. Gemini is also an explicit component of the final natural-language prototype and is disclosed as such.
