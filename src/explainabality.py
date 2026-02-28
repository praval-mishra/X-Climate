import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import joblib
import matplotlib.pyplot as plt

# Load data and models
df = pd.read_csv('data/processed_climate_data.csv')

features = ['T2M_MAX', 'T2M_MIN', 'T2M', 'RH2M', 'WS2M', 'PRECTOTCORR', 'MONTH']
class_names = ['Normal', 'Heatwave', 'Cold Wave', 'Heavy Rainfall']

X = df[features]
y = df['ANOMALY']

# Split same way as model.py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Load saved models
rf_model = joblib.load('models/random_forest.pkl')
gb_model = joblib.load('models/gradient_boosting.pkl')

# ─────────────────────────────────────────
# SHAP — Global Explanation (Random Forest)
# ─────────────────────────────────────────
print("Generating SHAP explanations...")

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot — Global Feature Importance
shap.summary_plot(shap_values, X_test, 
                  feature_names=features,
                  class_names=class_names,
                  plot_type="bar",
                  show=False)
plt.title("SHAP Global Feature Importance")
plt.tight_layout()
plt.savefig('outputs/shap_global_importance.png', dpi=150)
plt.close()
print("SHAP global plot saved.")

# SHAP Detail Plot for Heatwave class (class 1)
shap_vals_heatwave = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]

shap.summary_plot(shap_vals_heatwave, X_test,
                  feature_names=features,
                  show=False)
plt.title("SHAP Feature Impact — Heatwave Detection")
plt.tight_layout()
plt.savefig('outputs/shap_heatwave_detail.png', dpi=150)
plt.close()
print("SHAP heatwave detail plot saved.")
# ─────────────────────────────────────────
# LIME — Local Explanation
# ─────────────────────────────────────────
print("Generating LIME explanations...")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=features,
    class_names=class_names,
    mode='classification'
)

# Explain one anomaly instance — find first heatwave in test set
heatwave_indices = X_test[y_test == 1].index
instance = X_test.loc[heatwave_indices[0]].values

exp = lime_explainer.explain_instance(
    instance,
    rf_model.predict_proba,
    num_features=7
)

exp.save_to_file('outputs/lime_heatwave_explanation.html')
print("LIME explanation saved.")

print("\nAll explanations generated successfully.")