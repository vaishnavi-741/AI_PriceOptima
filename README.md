
Project Title
PriceOptima – AI-Powered Dynamic Pricing System

Project Objective

Price Optima provides automated price recommendations using historical sales data, real-time stock position, temporal variables, and retail market signals.
Users submit a few operational fields from the dashboard and the system applies feature-engineered attributes and machine learning models to compute demand and suggest an optimal selling price.

Dataset Description
Location: /dataset folder
Primary datasets used include files such as master_dataset_cleaned.csv and internally engineered variations.

Key columns observed in processing:
-	Cost price, Selling price
-	Units sold, Stock level
-	Day, Week, Month
-	Competitor price, Product category
-	Weekend/Holiday indicators
Datasets required extensive data cleaning because they contain missing values, inconsistent date formats, inventory anomalies, and seasonality patterns typical of fast-moving consumer goods.

Technologies Used :

-	Backend: Python, FastAPI, Uvicorn, Pydantic
-	Frontend: React.js, JavaScript, HTML/CSS
-	ML & Data: pandas, numpy, scikit-learn, LightGBM, joblib
-	Utilities & Dev: GitHub, npm, curl, virtual environments

Model Development Summary

Exploratory data insights highlighted metrics such as:
-	Daily revenue behavior
-	Demand elasticity with respect to price
-	Stock levels influencing lost sales
-	Weekend and seasonal effects

Key feature engineering performed:

-	Date transformations (day, week, month, weekend)
-	Categorical encoding
-	Rolling demand behavior
-	Ratio features: stock vs predicted demand
-	Competitor discount differentials

Multiple machine learning models were evaluated, including XGBoost and LightGBM.
The optimized model was serialized as priceoptima_lgb_model.joblib and deployed in the API.
Fallback logic is integrated if the trained file is unavailable.
Backend Implementation (FastAPI)
Entry File: app.py

Live API endpoints include:

1.	GET / – service heartbeat
2.	POST /predict – accepts JSON payload with fields:
3.	cost, stock, day, week, month, is_weekend,
4.	competitor, discount, holiday, last_week_sales, category

Returns:
{ "predicted_demand": x.xx,
  "recommended_price": x.xx }

CORS is enabled to allow communication with the React dashboard.
Model Loading Pattern:

import joblib
model = joblib.load('priceoptima_lgb_model.joblib')
Error handling ensures uninterrupted testing during UI development.
Dashboard Implementation (React.js)
Frontend Location: /dashboard/src/App.js

Dashboard delivers:

-	Input form for all required model variables
-	Submit button mapped to /predict POST request
-	Dynamic display of recommended price
-	Basic fail-safe display for API/validation errors

The UI is optimized for clarity so a business user can experiment interactively.
Key Outputs & Results
-	API responds with demand estimate and price suggestion
-	Dashboard renders results instantly
-	Internal project tests show meaningful pricing uplift patterns when simulated against historical data
-	Early EDA showed improvement opportunities in pricing intervention windows
Real-World Usage Context

Operations analysts can apply Price Optima daily:
1.	Enter cost, stock status and market context
2.	Receive recommended selling price
3.	Adjust price strategies in e-commerce/POS systems

Milestones
Milestone 1 - Requirements, scoping, and dataset selection
Milestone 2 - EDA, visualization, data distribution, cleaning
Milestone 3 - Feature engineering, temporal variables, competitor logic
Milestone 4 - Rule-based pricing baseline
Milestone 5 - ML-based pricing and model selection
Milestone 6 - API deployment, React dashboard, README documentation

Conclusion and Future Enhancements
Price Optima completes an end-to-end AI pricing workflow:
data preparation, feature generation, ML modeling, inference API, and interactive dashboard.
