# 🌍 X-Climate: Explainable AI for Climate Anomaly Detection

## Overview
X-Climate is a machine learning project that detects climate anomalies such as 
heatwaves, cold waves, and heavy rainfall events using ensemble learning models. 
It uses SHAP and LIME to explain model predictions, making the system transparent 
and trustworthy for real-world climate decision-making.

## Location & Data
- **Location:** Hyderabad, India (Lat: 17.385, Lon: 78.4867)
- **Period:** 2010–2023 (5113 daily records)
- **Source:** NASA POWER Climate Dataset

## Features Used
| Feature | Description |
|---|---|
| T2M_MAX | Maximum Temperature at 2m (°C) |
| T2M_MIN | Minimum Temperature at 2m (°C) |
| T2M | Average Temperature at 2m (°C) |
| RH2M | Relative Humidity at 2m (%) |
| WS2M | Wind Speed at 2m (m/s) |
| PRECTOTCORR | Corrected Precipitation (mm/day) |
| MONTH | Month extracted from date |

## Anomaly Classes
| Label | Class | Detection Rule |
|---|---|---|
| 0 | Normal | No significant deviation |
| 1 | Heatwave | T2M_MAX Z-score > 2 |
| 2 | Cold Wave | T2M_MIN Z-score < -2 |
| 3 | Heavy Rainfall | PRECTOTCORR Z-score > 2 |

## Models
- **Random Forest** — 97% accuracy
- **Gradient Boosting** — 98% accuracy

## Explainability
- **SHAP** — Global feature importance across all anomaly classes
- **LIME** — Local explanation for individual day predictions

## Project Structure
```
X-Climate/
├── data/
│   ├── hyderabad_climate.csv
│   └── processed_climate_data.csv
├── models/
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
├── outputs/
│   ├── shap_global_importance.png
│   ├── shap_heatwave_detail.png
│   └── lime_heatwave_explanation.html
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── explainability.py
│   └── dashboard.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess data
```bash
python src/preprocessing.py
```

### 3. Train models
```bash
python src/model.py
```

### 4. Generate SHAP and LIME explanations
```bash
python src/explainability.py
```

### 5. Launch dashboard
```bash
streamlit run src/dashboard.py
```

## Tech Stack
Python, Scikit-learn, SHAP, LIME, Streamlit, Pandas, Matplotlib

## Author
Praval Mishra | B.Tech CSE (Data Science), Swami Vivekananda Institute of Technology | 3rd

