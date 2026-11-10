# AI Usage Disclosure

**Project:** Bike Sharing Demand Intelligence  
**Case Study:** Sigmoid Data Science Trainee  
**Approach:** Human-led, AI-assisted

## 1. My Contribution

I led the project from problem definition and data exploration to model evaluation, prototype development and the final business recommendation.

My contributions included:
- Performing data quality checks and exploratory analysis.
- Formulating and evaluating the commute-window hypothesis.
- Developing and evaluating the Random Forest demand prediction model.
- Analysing model errors, feature importance and fleet-sizing scenarios.
- Developing and testing the Streamlit dashboard and Gemini-powered Business Assistant.
- Interpreting findings and recommending a controlled redistribution pilot before fleet expansion.

## 2. AI Tools Used

**ChatGPT (OpenAI):** Used for learning concepts, exploring analytical approaches, coding assistance, debugging, interpreting outputs, test development and documentation.

**Google Gemini API:** Integrated into the Business Assistant to understand natural-language questions and select predefined analytical functions. Calculations are performed locally using Python and the project data, with validation and fallback mechanisms.

## 3. Validation and Responsibility

I ran the notebooks, reviewed analytical outputs, tested the application and evaluated the final recommendations. AI-generated code and suggestions were treated as assistance and checked before use.

The final recorded Random Forest metrics were MAE 64.09, RMSE 94.54 and R² 0.8068.

**Testing results:** 27 existing project tests and 10 Business Assistant tests passed, giving 37 passing tests with one non-blocking warning. A live Gemini function-calling response was also verified separately.

## 4. Limitations and Disclosure

The analysis is based on completed rentals and cannot establish actual station-level shortages or unmet demand. Fleet estimates are assumption-based scenarios, not verified requirements.

The Gemini API key is stored outside source code using environment variables or Streamlit secrets.

AI tools supported development, but I remain responsible for the project's analytical decisions, implementation, validation and final conclusions.
