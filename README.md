# Sales & Demand Forecasting

A machine learning project that predicts future retail sales using historical data. Built as part of a business intelligence task to help store owners and managers make better inventory, staffing, and cash flow decisions.

-----

## What It Does

The script generates 3 years of realistic retail sales data, trains three forecasting models, evaluates them, and produces a 60-day forward forecast with confidence bands. Everything is exported as charts and CSVs that are easy to share with non-technical stakeholders.

-----

## Project Structure

```
task-1-sales-forecasting/
├── sales_forecasting.py
├── requirements.txt
├── README.md
└── outputs/
    ├── sales_forecasting_dashboard.png
    ├── feature_importance.png
    ├── forecast_data.csv
    └── model_metrics.csv
```

-----

## Setup

```bash
pip install -r requirements.txt
python sales_forecasting.py
```

That’s it. All outputs are saved to the `outputs/` folder automatically.

-----

## How It Works

### 1. Data

Synthetic daily sales data from 2022–2024 with realistic patterns baked in — weekend spikes, holiday surges (Nov–Dec), a summer dip, and one-off promo events like Black Friday and Valentine’s Day.

### 2. Feature Engineering

Extracts 30+ features from the date column including:

- Calendar features (day of week, month, quarter)
- Fourier terms to capture weekly and annual seasonality
- Lag features (sales from 7, 14, 21, 28 days ago)
- Rolling mean and standard deviation (7/14/30-day windows)

### 3. Models Trained

Three models are trained and compared on a 90-day holdout test set:

|Model               |MAE   |RMSE  |R²        |MAPE     |
|--------------------|------|------|----------|---------|
|**Ridge Regression**|$38.35|$49.74|**0.7308**|**4.82%**|
|Gradient Boosting   |$48.44|$61.21|0.5922    |6.07%    |
|Random Forest       |$57.81|$74.39|0.3978    |7.09%    |

Ridge Regression came out on top — it generalises well when the feature set already captures the main patterns (which the Fourier and lag features do here).

### 4. Forecast

The best model rolls forward 60 days iteratively, using its own predictions to fill lag features. A confidence band (±1.5 standard deviations) is added around the forecast to show the expected range.

-----

## Outputs

**`sales_forecasting_dashboard.png`** — A 5-panel visual covering sales history, day-of-week and monthly seasonality, model comparison, actual vs predicted, and the 60-day forecast.

**`feature_importance.png`** — Shows which features drive the predictions most (for tree-based models).

**`forecast_data.csv`** — Daily forecast values with lower and upper confidence bounds.

**`model_metrics.csv`** — MAE, RMSE, R², and MAPE for all three models side by side.

-----

## Business Takeaways

- **Inventory**: buffer ~15–20% above the daily forecast average to handle demand spikes
- **Staffing**: weekends consistently outperform weekdays by 40–50%, plan shifts accordingly
- **Cash flow**: the 60-day total projection gives a concrete number to plan procurement around
- **Seasonality**: November and December are peak months — stock and staff up early

-----

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```
