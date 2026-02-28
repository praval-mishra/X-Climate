import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load processed data
df = pd.read_csv('data/processed_climate_data.csv')

# Define features and target
features = ['T2M_MAX', 'T2M_MIN', 'T2M', 'RH2M', 'WS2M', 'PRECTOTCORR',
             'MONTH']

X = df[features]
y = df['ANOMALY']

# Split data — 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")
print(classification_report(y_test, rf_preds))

# Train Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)

print("\n--- Gradient Boosting Results ---")
print(classification_report(y_test, gb_preds))

# Save both models
joblib.dump(rf_model, 'models/random_forest.pkl')
joblib.dump(gb_model, 'models/gradient_boosting.pkl')
print("\nModels saved successfully.")