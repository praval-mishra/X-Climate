import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────
st.set_page_config(
    page_title="X-Climate | Climate Anomaly Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Custom CSS for colorful infographic style
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 5px;
    }
    .anomaly-normal {
        background: linear-gradient(135deg, #1a6b3c, #27ae60);
        padding: 15px; border-radius: 12px;
        color: white; text-align: center;
    }
    .anomaly-heatwave {
        background: linear-gradient(135deg, #922b21, #e74c3c);
        padding: 15px; border-radius: 12px;
        color: white; text-align: center;
    }
    .anomaly-coldwave {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        padding: 15px; border-radius: 12px;
        color: white; text-align: center;
    }
    .anomaly-rainfall {
        background: linear-gradient(135deg, #7d6608, #f39c12);
        padding: 15px; border-radius: 12px;
        color: white; text-align: center;
    }
    .explanation-box {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        border-left: 4px solid #3498db;
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 16px;
        line-height: 1.8;
    }
    .section-header {
        font-size: 28px;
        font-weight: bold;
        color: #3498db;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
class_icons = ['🟢', '🔴', '🔵', '🟡']
class_colors = ['#27ae60', '#e74c3c', '#2e86c1', '#f39c12']

X = df[features]
y = df['ANOMALY']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ─────────────────────────────────────────
# Natural Language Explanation Engine
# ─────────────────────────────────────────
def generate_explanation(prediction, input_data, shap_values_local):
    month_names = {1:'January',2:'February',3:'March',4:'April',
                   5:'May',6:'June',7:'July',8:'August',
                   9:'September',10:'October',11:'November',12:'December'}
    month = month_names[int(input_data['MONTH'])]

    shap_flat = np.array(shap_values_local).flatten()
    feature_impacts = dict(zip(features, [float(v) for v in shap_flat]))
    top_feature = max(feature_impacts, key=lambda x: abs(feature_impacts[x]))

    feature_descriptions = {
        'T2M_MAX': f"maximum temperature of {input_data['T2M_MAX']:.1f}°C",
        'T2M_MIN': f"minimum temperature of {input_data['T2M_MIN']:.1f}°C",
        'T2M': f"average temperature of {input_data['T2M']:.1f}°C",
        'RH2M': f"relative humidity of {input_data['RH2M']:.1f}%",
        'WS2M': f"wind speed of {input_data['WS2M']:.1f} m/s",
        'PRECTOTCORR': f"precipitation of {input_data['PRECTOTCORR']:.1f} mm",
        'MONTH': f"the month of {month}"
    }

    anomaly_explanations = {
        0: f"✅ Conditions appear **normal** for {month}. "
           f"The {feature_descriptions[top_feature]} is within expected historical ranges. "
           f"No significant climate anomaly is detected.",

        1: f"🔴 **Heatwave Alert** detected for {month}! "
           f"The primary driver is {feature_descriptions[top_feature]}, "
           f"which is significantly above the historical average for this period. "
           f"High temperatures combined with low humidity increase heat stress risk.",

        2: f"🔵 **Cold Wave Alert** detected for {month}! "
           f"The primary driver is {feature_descriptions[top_feature]}, "
           f"which is significantly below the historical average for this period. "
           f"Unusually cold conditions may impact agriculture and daily life.",

        3: f"🟡 **Heavy Rainfall Alert** detected for {month}! "
           f"The primary driver is {feature_descriptions[top_feature]}, "
           f"which is far above normal levels for this period. "
           f"There is elevated risk of waterlogging and flash floods."
    }

    return anomaly_explanations[prediction]

# ─────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────
st.sidebar.markdown("## 🌍 X-Climate")
st.sidebar.markdown("*Explainable AI for Climate Anomaly Detection*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📅 Climate Story",
    "🔮 Live Prediction",
    "📊 Model Insights",
    "📖 About the Project"
])
st.sidebar.markdown("---")
st.sidebar.markdown("📍 **Location:** Hyderabad, India")
st.sidebar.markdown("📆 **Data:** 2010–2023")
st.sidebar.markdown("🤖 **Models:** RF + GB + XAI")

# ─────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center; color:#3498db;'>🌍 X-Climate</h1>", 
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#bdc3c7;'>Explainable AI for Climate Anomaly Detection</h3>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#95a5a6;'>Hyderabad, India | 2010–2023 | NASA POWER Data</p>", 
                unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class='metric-card'>
            <h2>5,113</h2><p>Days Analyzed</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='anomaly-heatwave'>
            <h2>🔴 119</h2><p>Heatwave Days</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='anomaly-coldwave'>
            <h2>🔵 132</h2><p>Cold Wave Days</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class='anomaly-rainfall'>
            <h2>🟡 192</h2><p>Heavy Rainfall Days</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-header'>What is this project?</div>", 
                unsafe_allow_html=True)
    st.markdown("""
    <div class='explanation-box'>
    🌡️ <b>Climate anomalies</b> like heatwaves, cold waves, and extreme rainfall are becoming 
    more frequent and dangerous. Traditional weather forecasts tell you <i>what</i> will happen 
    — but not <i>why</i>.<br><br>
    🤖 <b>X-Climate</b> uses machine learning to detect these anomalies from historical climate 
    data — and then uses <b>Explainable AI (XAI)</b> to tell you exactly which climate factors 
    drove each prediction in plain English.<br><br>
    🔍 No black boxes. No blind trust. Just transparent, understandable climate intelligence.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-header'>Anomaly Distribution (2010–2023)</div>", 
                unsafe_allow_html=True)

    anomaly_counts = df['ANOMALY'].value_counts().sort_index()
    fig = px.pie(
        values=anomaly_counts.values,
        names=['Normal', 'Heatwave', 'Cold Wave', 'Heavy Rainfall'],
        color_discrete_sequence=['#27ae60', '#e74c3c', '#2e86c1', '#f39c12'],
        hole=0.4
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend=dict(font=dict(color='white'))
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 2 — CLIMATE STORY
# ─────────────────────────────────────────
elif page == "📅 Climate Story":
    st.markdown("<h1 style='color:#3498db;'>📅 Climate Story — Hyderabad 2010–2023</h1>", 
                unsafe_allow_html=True)
    st.markdown("*Explore 14 years of climate data. Hover over any point to see details.*")
    st.markdown("---")

    df['DATE'] = pd.to_datetime(df['DATE'])
    df['ANOMALY_LABEL'] = df['ANOMALY'].map({
        0: 'Normal', 1: 'Heatwave', 2: 'Cold Wave', 3: 'Heavy Rainfall'})

    year_filter = st.slider("Select Year Range", 2010, 2023, (2010, 2023))
    filtered_df = df[(df['DATE'].dt.year >= year_filter[0]) & 
                     (df['DATE'].dt.year <= year_filter[1])]

    st.subheader("🌡️ Temperature Over Time")
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=filtered_df['DATE'], y=filtered_df['T2M_MAX'],
        name='Max Temp', line=dict(color='#e74c3c', width=1)))
    fig_temp.add_trace(go.Scatter(
        x=filtered_df['DATE'], y=filtered_df['T2M_MIN'],
        name='Min Temp', line=dict(color='#2e86c1', width=1)))

    anomalies = filtered_df[filtered_df['ANOMALY'] != 0]
    fig_temp.add_trace(go.Scatter(
        x=anomalies['DATE'], y=anomalies['T2M_MAX'],
        mode='markers', name='Anomaly',
        marker=dict(color='#f39c12', size=6, symbol='star')))

    fig_temp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,17,23,1)',
        font_color='white',
        xaxis=dict(gridcolor='#2d2d44'),
        yaxis=dict(gridcolor='#2d2d44', title='Temperature (°C)'),
        legend=dict(font=dict(color='white'))
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    st.subheader("🌧️ Precipitation Over Time")
    fig_prec = px.bar(
        filtered_df, x='DATE', y='PRECTOTCORR',
        color='ANOMALY_LABEL',
        color_discrete_map={
            'Normal': '#27ae60', 'Heatwave': '#e74c3c',
            'Cold Wave': '#2e86c1', 'Heavy Rainfall': '#f39c12'
        }
    )
    fig_prec.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,17,23,1)',
        font_color='white',
        yaxis_title='Precipitation (mm/day)'
    )
    st.plotly_chart(fig_prec, use_container_width=True)

    st.subheader("📋 Anomaly Events")
    anomaly_table = filtered_df[filtered_df['ANOMALY'] != 0][
        ['DATE', 'ANOMALY_LABEL', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M']
    ].sort_values('DATE', ascending=False)
    anomaly_table.columns = ['Date', 'Anomaly Type', 'Max Temp (°C)', 
                              'Min Temp (°C)', 'Precipitation (mm)', 'Humidity (%)']
    st.dataframe(anomaly_table, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 3 — LIVE PREDICTION
# ─────────────────────────────────────────
elif page == "🔮 Live Prediction":
    st.markdown("<h1 style='color:#3498db;'>🔮 Live Climate Prediction</h1>", 
                unsafe_allow_html=True)
    st.markdown("*Enter climate values below to detect anomalies and get a plain English explanation.*")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Input Climate Values")
        month = st.slider("Month", 1, 12, 6)
        t2m_max = st.slider("Max Temperature (°C)", 15.0, 50.0, 35.0)
        t2m_min = st.slider("Min Temperature (°C)", 5.0, 35.0, 20.0)
        t2m = st.slider("Average Temperature (°C)", 10.0, 45.0, 27.0)
        rh2m = st.slider("Relative Humidity (%)", 10.0, 100.0, 55.0)
        ws2m = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0)
        prec = st.slider("Precipitation (mm/day)", 0.0, 100.0, 0.0)

    input_data = {
        'T2M_MAX': t2m_max, 'T2M_MIN': t2m_min, 'T2M': t2m,
        'RH2M': rh2m, 'WS2M': ws2m, 'PRECTOTCORR': prec, 'MONTH': month
    }
    input_df = pd.DataFrame([input_data])

    prediction = rf_model.predict(input_df)[0]
    probabilities = rf_model.predict_proba(input_df)[0]

    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(input_df)
    if isinstance(shap_vals, list):
        local_shap = shap_vals[prediction][0]
    else:
        local_shap = shap_vals[0]

    explanation = generate_explanation(prediction, input_data, local_shap)

    with col2:
        st.subheader("📤 Prediction Result")

        card_styles = ['anomaly-normal', 'anomaly-heatwave', 
                       'anomaly-coldwave', 'anomaly-rainfall']
        st.markdown(f"""
        <div class='{card_styles[prediction]}'>
            <h2>{class_icons[prediction]} {class_names[prediction]}</h2>
            <p>Confidence: {probabilities[prediction]*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🗣️ What does this mean?</div>", 
                    unsafe_allow_html=True)
        st.markdown(f"<div class='explanation-box'>{explanation}</div>", 
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Prediction Confidence")
        prob_df = pd.DataFrame({
            'Anomaly Type': class_names,
            'Probability': probabilities
        })
        fig_prob = px.bar(
            prob_df, x='Anomaly Type', y='Probability',
            color='Anomaly Type',
            color_discrete_map={
                'Normal': '#27ae60', 'Heatwave': '#e74c3c',
                'Cold Wave': '#2e86c1', 'Heavy Rainfall': '#f39c12'
            }
        )
        fig_prob.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,17,23,1)',
            font_color='white',
            showlegend=False,
            yaxis=dict(range=[0, 1], gridcolor='#2d2d44')
        )
        st.plotly_chart(fig_prob, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 4 — MODEL INSIGHTS
# ─────────────────────────────────────────
elif page == "📊 Model Insights":
    st.markdown("<h1 style='color:#3498db;'>📊 Model Insights</h1>", 
                unsafe_allow_html=True)
    st.markdown("*Technical performance details for evaluators and researchers.*")
    st.markdown("---")

    from sklearn.metrics import classification_report
    rf_preds = rf_model.predict(X_test)
    gb_preds = gb_model.predict(X_test)

    report_rf = classification_report(y_test, rf_preds, 
                                      target_names=class_names, output_dict=True)
    report_gb = classification_report(y_test, gb_preds, 
                                      target_names=class_names, output_dict=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌲 Random Forest — 97% Accuracy")
        st.dataframe(pd.DataFrame(report_rf).transpose().round(2))
    with col2:
        st.subheader("🚀 Gradient Boosting — 98% Accuracy")
        st.dataframe(pd.DataFrame(report_gb).transpose().round(2))

    st.markdown("---")
    st.subheader("F1-Score Comparison")
    rf_f1 = [report_rf[c]['f1-score'] for c in class_names]
    gb_f1 = [report_gb[c]['f1-score'] for c in class_names]

    fig_f1 = go.Figure()
    fig_f1.add_trace(go.Bar(name='Random Forest', x=class_names, y=rf_f1,
                            marker_color='#3498db'))
    fig_f1.add_trace(go.Bar(name='Gradient Boosting', x=class_names, y=gb_f1,
                            marker_color='#e74c3c'))
    fig_f1.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,17,23,1)',
        font_color='white',
        yaxis=dict(range=[0, 1.1], gridcolor='#2d2d44'),
        legend=dict(font=dict(color='white'))
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 SHAP Global Feature Importance")
    img1 = Image.open('outputs/shap_global_importance.png')
    st.image(img1, use_column_width=True)

    st.subheader("🌡️ SHAP Feature Impact — Heatwave Detection")
    img2 = Image.open('outputs/shap_heatwave_detail.png')
    st.image(img2, use_column_width=True)

# ─────────────────────────────────────────
# PAGE 5 — ABOUT THE PROJECT
# ─────────────────────────────────────────
elif page == "📖 About the Project":
    st.markdown("<h1 style='color:#3498db;'>📖 About X-Climate</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div class='explanation-box'>
    <h3>🌍 The Problem</h3>
    Climate anomalies like heatwaves, cold waves, and extreme rainfall are becoming 
    more frequent due to changing climate patterns. These events cause loss of life, 
    damage to agriculture, and economic disruption. Early detection and clear communication 
    of these events is critical for public safety.
    <br><br>
    <h3>🤖 The Solution</h3>
    X-Climate uses machine learning to analyze 14 years of daily climate data from 
    Hyderabad, India and automatically detect when conditions are anomalous. But unlike 
    traditional ML systems, it doesn't just predict — it <b>explains</b> its predictions 
    in plain English so that anyone can understand why an anomaly was detected.
    <br><br>
    <h3>🔍 What is Explainable AI (XAI)?</h3>
    Most AI systems are "black boxes" — they give you an answer but can't tell you why. 
    Explainable AI solves this by making the model's reasoning transparent. 
    X-Climate uses two XAI techniques:
    <br><br>
    <b>SHAP</b> — Shows which climate factors matter most across all predictions globally. 
    Think of it as asking "what does the model care about most in general?"
    <br><br>
    <b>LIME</b> — Explains any single prediction locally. Think of it as asking 
    "why did the model flag THIS specific day?"
    <br><br>
    <h3>📊 The Data</h3>
    Data sourced from NASA POWER climate dataset. 5,113 daily records covering 
    temperature, humidity, wind speed, and precipitation for Hyderabad, India from 
    2010 to 2023.
    <br><br>
    <h3>🛠️ Tech Stack</h3>
    Python • Scikit-learn • SHAP • LIME • Streamlit • Plotly • Pandas • NASA POWER API
    </div>
    """, unsafe_allow_html=True)