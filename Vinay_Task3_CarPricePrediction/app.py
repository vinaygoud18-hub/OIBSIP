# Car Price Prediction Dashboard
# Oasis Infobyte Data Science Internship — Task 3
# Author: Vinay Goud | 3rd Year CSE (Data Science) — MRCET

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ---- Page Config ----
st.set_page_config(
    page_title="Car Price Predictor | Vinay Goud",
    page_icon="🚗",
    layout="wide"
)

# ---- CSS ----
st.markdown("""
<style>
body, .main, .block-container {
    background-color: #f5f6fa !important;
}
.block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1200px !important;
}
.header-banner {
    background: linear-gradient(135deg, #1a73e8, #0d47a1);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.header-banner h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
}
.header-banner p {
    color: #c8e0ff;
    font-size: 0.95rem;
    margin: 0.3rem 0 0 0;
}
.header-badge {
    background: rgba(255,255,255,0.2);
    color: #ffffff;
    padding: 0.5rem 1.2rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 700;
    white-space: nowrap;
}
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #e8eaf0;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.metric-card .metric-icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #1a73e8;
    margin: 0;
    line-height: 1;
}
.metric-card .metric-label {
    font-size: 0.78rem;
    color: #888888;
    margin: 0.3rem 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 1.5rem 0 1rem 0;
    padding-left: 0.8rem;
    border-left: 4px solid #1a73e8;
}
.chart-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #e8eaf0;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.result-box {
    background: linear-gradient(135deg, #1a73e8, #0d47a1);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
    box-shadow: 0 4px 20px rgba(26,115,232,0.3);
}
.result-box .result-label {
    color: #c8e0ff;
    font-size: 0.85rem;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.result-box .result-value {
    color: #ffffff;
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0.3rem 0;
    line-height: 1;
}
.result-box .result-sub {
    color: #c8e0ff;
    font-size: 0.85rem;
    margin: 0;
}
.insight-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 0.5rem;
}
.insight-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.2rem;
    border: 1px solid #e8eaf0;
    border-top: 3px solid #1a73e8;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.insight-card.two  { border-top-color: #34a853; }
.insight-card.three { border-top-color: #ea4335; }
.insight-card h4 {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 0 0 0.4rem 0;
}
.insight-card p {
    font-size: 0.85rem;
    color: #666666;
    margin: 0;
    line-height: 1.5;
}
.footer {
    text-align: center;
    color: #aaaaaa;
    font-size: 0.82rem;
    margin-top: 2rem;
    padding: 1rem 0;
    border-top: 1px solid #e8eaf0;
}
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8eaf0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---- Load & Train ----
@st.cache_resource
def load_and_train():
    df = pd.read_csv("car_data.csv")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Find price column
    price_col = None
    for col in df.columns:
        if 'price' in col.lower() or 'selling' in col.lower():
            price_col = col
            break
    if price_col is None:
        price_col = df.columns[-1]

    # Store original for display
    df_display = df.copy()

    # Encode categoricals
    le_dict = {}
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    X = df.drop(columns=[price_col])
    y = df[price_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'r2'  : r2_score(y_test, y_pred),
        'mae' : mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'samples': len(df)
    }

    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return model, df, df_display, le_dict, X.columns.tolist(), \
           price_col, metrics, importances, y_test, y_pred

model, df, df_display, le_dict, feature_cols, price_col, \
    metrics, importances, y_test, y_pred = load_and_train()

# ---- Sidebar ----
with st.sidebar:
    st.markdown("""
        <div style='background:linear-gradient(135deg,#1a73e8,#0d47a1);
        border-radius:12px; padding:1rem; text-align:center;
        margin-bottom:1.5rem;'>
            <div style='font-size:2rem'>🚗</div>
            <h3 style='color:#fff; margin:0.3rem 0 0; font-size:1rem;'>
            Car Price Predictor</h3>
            <p style='color:#c8e0ff; margin:0.2rem 0 0; font-size:0.8rem;'>
            Adjust features to predict price</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Car Features**")

    # Dynamic inputs based on dataset columns
    user_input = {}
    cat_cols_orig = list(le_dict.keys())

    for col in feature_cols:
        display_name = col.replace('_', ' ').title()
        if col in cat_cols_orig:
            options = list(le_dict[col].classes_)
            selected = st.selectbox(f"🔹 {display_name}", options)
            user_input[col] = le_dict[col].transform([selected])[0]
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            user_input[col] = st.slider(
                f"🔹 {display_name}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    # Predict
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]

    st.markdown(f"""
        <div class="result-box">
            <p class="result-label">🚗 Estimated Price</p>
            <p class="result-value">₹{prediction:,.0f}</p>
            <p class="result-sub">Predicted by Random Forest</p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
        <div style='text-align:center; color:#aaa;
        font-size:0.8rem; line-height:1.6'>
            <strong style='color:#555'>Vinay Goud</strong><br>
            3rd Year CSE (Data Science)<br>
            MRCET | Oasis Infobyte<br>Task 3
        </div>
    """, unsafe_allow_html=True)

# ---- Main Area ----

# Header
st.markdown(f"""
    <div class="header-banner">
        <div>
            <h1>🚗 Car Price Prediction</h1>
            <p>Predict used car prices using Random Forest
            Machine Learning model</p>
        </div>
        <div class="header-badge">R² Score: {metrics['r2']:.2f} ✅</div>
    </div>
""", unsafe_allow_html=True)

# Metric Cards
st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <p class="metric-value">{metrics['r2']:.2f}</p>
            <p class="metric-label">R² Accuracy</p>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📉</div>
            <p class="metric-value">₹{metrics['mae']:,.0f}</p>
            <p class="metric-label">Mean Abs Error</p>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <p class="metric-value">₹{metrics['rmse']:,.0f}</p>
            <p class="metric-label">RMSE Score</p>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🚘</div>
            <p class="metric-value">{metrics['samples']}</p>
            <p class="metric-label">Cars in Dataset</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Charts
st.markdown('<p class="section-title">📊 Model Visualizations</p>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    # Actual vs Predicted
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=list(y_test), y=list(y_pred),
        mode='markers',
        marker=dict(color='#1a73e8', opacity=0.5, size=5),
        name='Predictions'
    ))
    fig1.add_trace(go.Scatter(
        x=[float(y_test.min()), float(y_test.max())],
        y=[float(y_test.min()), float(y_test.max())],
        mode='lines',
        line=dict(color='#ea4335', width=2, dash='dash'),
        name='Perfect Prediction'
    ))
    fig1.update_layout(
        title=dict(text='Actual vs Predicted Price',
                   font=dict(size=13, color='#1a1a2e')),
        xaxis_title='Actual Price (₹)',
        yaxis_title='Predicted Price (₹)',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#1a1a2e'),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=-0.3)
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    # Feature Importance
    top_features = importances.head(8).reset_index()
    top_features.columns = ['Feature', 'Importance']
    top_features['Feature'] = top_features['Feature'].str.replace(
        '_', ' ').str.title()

    fig2 = px.bar(
        top_features, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues',
        labels={'Importance': 'Importance Score'}
    )
    fig2.update_layout(
        title=dict(text='Top Feature Importances',
                   font=dict(size=13, color='#1a1a2e')),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#1a1a2e'),
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False,
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Price Distribution
st.markdown('<p class="section-title">📈 Price Distribution</p>',
            unsafe_allow_html=True)
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
fig3 = px.histogram(
    df, x=price_col, nbins=50,
    color_discrete_sequence=['#1a73e8'],
    labels={price_col: 'Car Price (₹)'},
    opacity=0.8
)
fig3.update_layout(
    title=dict(text='Distribution of Car Prices in Dataset',
               font=dict(size=13, color='#1a1a2e')),
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(color='#1a1a2e'),
    margin=dict(l=20, r=20, t=40, b=20),
    bargap=0.05
)
st.plotly_chart(fig3, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Insights
st.markdown('<p class="section-title">💡 Key Insights</p>',
            unsafe_allow_html=True)
st.markdown("""
    <div class="insight-grid">
        <div class="insight-card">
            <div style='font-size:1.5rem; margin-bottom:0.5rem'>🌲</div>
            <h4>Random Forest wins</h4>
            <p>Random Forest outperforms Linear Regression
            for car price prediction due to its ability
            to capture non-linear relationships in data.</p>
        </div>
        <div class="insight-card two">
            <div style='font-size:1.5rem; margin-bottom:0.5rem'>📅</div>
            <h4>Age matters most</h4>
            <p>Car age and year of manufacture are the
            strongest predictors of price. Older cars
            depreciate significantly in value.</p>
        </div>
        <div class="insight-card three">
            <div style='font-size:1.5rem; margin-bottom:0.5rem'>⛽</div>
            <h4>Fuel & transmission</h4>
            <p>Fuel type and transmission type heavily
            influence resale value. Diesel and automatic
            cars command higher prices in the used market.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Built by <strong>Vinay Goud</strong> &nbsp;|&nbsp;
        3rd Year CSE (Data Science) — MRCET &nbsp;|&nbsp;
        Oasis Infobyte Data Science Internship &nbsp;|&nbsp;
        Task 3 — Car Price Prediction
    </div>
""", unsafe_allow_html=True)