import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image

# ─────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────
st.set_page_config(
    page_title="X-Climate: Anomaly Detection",
    page_icon="🌍",
    layout="wide"
)

# ─────────────────────────────────────────
# Load Data and Models
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data/processed_climate_data.csv')

@st.cache_resource
def load_models():
    rf = joblib.load('models/random_forest.pkl')
    gb = joblib.load('models/gradient_boosting.pkl')
    return rf, gb

df = load_data()
rf_model, gb_model = load_models()

features = ['T2M_MAX', 'T2M_MIN', 'T2M', 'RH2M', 'WS2M', 'PRECTOTCORR', 'MONTH']
class_names = ['Normal', 'Heatwave', 'Cold Wave', 'Heavy Rainfall']
anomaly_colors = {0: '🟢', 1: '🔴', 2: '🔵', 3: '🟡'}

X = df[features]
y = df['ANOMALY']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ─────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────
st.sidebar.title("🌍 X-Climate")
st.sidebar.markdown("**Explainable AI for Climate Anomaly Detection**")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Model Performance", 
    "SHAP Explanations",
    "LIME Explanation"
])

# ─────────────────────────────────────────
# Page 1 — Overview
# ─────────────────────────────────────────
if page == "Overview":
    st.title("Climate Anomaly Detection — Overview")
    st.markdown("**Location:** Hyderabad, India | **Period:** 2010–2023 | **Model:** Random Forest + Gradient Boosting + XAI")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Days", len(df))
    col2.metric("Heatwave Days", len(df[df['ANOMALY']==1]))
    col3.metric("Cold Wave Days", len(df[df['ANOMALY']==2]))
    col4.metric("Heavy Rainfall Days", len(df[df['ANOMALY']==3]))
    
    st.subheader("Anomaly Distribution")
    anomaly_counts = df['ANOMALY'].value_counts().sort_index()
    anomaly_labels = ['Normal', 'Heatwave', 'Cold Wave', 'Heavy Rainfall']
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(anomaly_labels, anomaly_counts.values, 
                  color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])
    ax.set_ylabel("Number of Days")
    ax.set_title("Distribution of Climate Anomalies (2010-2023)")
    for bar, val in zip(bars, anomaly_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(val), ha='center', fontweight='bold')
    st.pyplot(fig)
    plt.close()
    
    st.subheader("Raw Climate Data Sample")
    st.dataframe(df[['DATE', 'T2M_MAX', 'T2M_MIN', 'RH2M', 
                      'PRECTOTCORR', 'WS2M', 'ANOMALY']].head(20))

# ─────────────────────────────────────────
# Page 2 — Model Performance
# ─────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Random Forest")
        rf_preds = rf_model.predict(X_test)
        report_rf = classification_report(y_test, rf_preds, 
                                          target_names=class_names, 
                                          output_dict=True)
        st.dataframe(pd.DataFrame(report_rf).transpose().round(2))
    
    with col2:
        st.subheader("Gradient Boosting")
        gb_preds = gb_model.predict(X_test)
        report_gb = classification_report(y_test, gb_preds, 
                                          target_names=class_names, 
                                          output_dict=True)
        st.dataframe(pd.DataFrame(report_gb).transpose().round(2))
    
    st.subheader("F1-Score Comparison")
    rf_f1 = [report_rf[c]['f1-score'] for c in class_names]
    gb_f1 = [report_gb[c]['f1-score'] for c in class_names]
    
    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, rf_f1, 0.4, label='Random Forest', color='#3498db')
    ax.bar(x + 0.2, gb_f1, 0.4, label='Gradient Boosting', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score Comparison per Anomaly Class')
    ax.legend()
    ax.set_ylim(0, 1.1)
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────
# Page 3 — SHAP Explanations
# ─────────────────────────────────────────
elif page == "SHAP Explanations":
    st.title("SHAP — Global Explainability")
    st.markdown("SHAP shows which features drive anomaly predictions globally across the entire dataset.")
    
    st.subheader("Global Feature Importance (All Classes)")
    img1 = Image.open('outputs/shap_global_importance.png')
    st.image(img1, use_column_width=True)
    
    st.subheader("Feature Impact on Heatwave Detection")
    img2 = Image.open('outputs/shap_heatwave_detail.png')
    st.image(img2, use_column_width=True)
    
    st.markdown("""
    **How to read the heatwave plot:**
    - Features are ranked by importance (top = most important)
    - Red dots = high feature value, Blue dots = low feature value  
    - Positive SHAP value = pushes toward Heatwave prediction
    - Negative SHAP value = pushes away from Heatwave prediction
    """)

# ─────────────────────────────────────────
# Page 4 — LIME Explanation
# ─────────────────────────────────────────
elif page == "LIME Explanation":
    st.title("LIME — Local Explainability")
    st.markdown("LIME explains why the model made a specific prediction for any individual day.")
    
    st.subheader("Select a date to explain")
    
    date_list = df['DATE'].tolist()
    selected_date = st.selectbox("Choose a date", date_list)
    
    selected_row = df[df['DATE'] == selected_date][features].values[0]
    actual_label = df[df['DATE'] == selected_date]['ANOMALY'].values[0]
    
    st.info(f"Actual label: **{class_names[actual_label]}**")
    
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=features,
        class_names=class_names,
        mode='classification'
    )
    
    exp = lime_explainer.explain_instance(
        selected_row,
        rf_model.predict_proba,
        num_features=7
    )
    
    prediction_probs = rf_model.predict_proba(selected_row.reshape(1, -1))[0]
    
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Class': class_names,
        'Probability': prediction_probs.round(3)
    })
    st.dataframe(prob_df)
    
    st.subheader("Feature Contributions")
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)
    plt.close()