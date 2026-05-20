# Sales Prediction App - Oasis Infobyte Internship Task 5
# Author: Vinay Goud | 3rd Year CSE (Data Science) — MRCET

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Sales Predictor | Vinay Goud",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
/* ---- Reset & Base ---- */
body, .main, .block-container {
    background-color: #f5f6fa !important;
}
.block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1200px !important;
}

/* ---- Header Banner ---- */
.header-banner {
    background: #1a73e8;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
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
    background: rgba(255,255,255,0.15);
    color: #ffffff;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    white-space: nowrap;
}

/* ---- Metric Cards ---- */
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
}
.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #1a73e8;
    margin: 0;
    line-height: 1;
}
.metric-card .metric-label {
    font-size: 0.8rem;
    color: #888888;
    margin: 0.4rem 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-card .metric-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

/* ---- Section Title ---- */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e8eaf0;
    margin-left: 0.5rem;
}

/* ---- Prediction Card ---- */
.predict-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #e8eaf0;
    height: 100%;
}
.result-display {
    background: #1a73e8;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.result-display .result-label {
    color: #c8e0ff;
    font-size: 0.85rem;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.result-display .result-value {
    color: #ffffff;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0.2rem 0;
    line-height: 1;
}
.result-display .result-sub {
    color: #c8e0ff;
    font-size: 0.85rem;
    margin: 0;
}

/* ---- Budget Breakdown ---- */
.budget-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 0.9rem;
}
.budget-row:last-child { border-bottom: none; }
.budget-label { color: #555555; }
.budget-value {
    font-weight: 700;
    color: #1a73e8;
}

/* ---- Chart Card ---- */
.chart-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #e8eaf0;
    margin-bottom: 1rem;
}

/* ---- Insight Cards ---- */
.insight-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}
.insight-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.2rem;
    border: 1px solid #e8eaf0;
    border-top: 3px solid #1a73e8;
}
.insight-card.radio { border-top-color: #34a853; }
.insight-card.newspaper { border-top-color: #ea4335; }
.insight-card .insight-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
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

/* ---- Footer ---- */
.footer {
    text-align: center;
    color: #aaaaaa;
    font-size: 0.82rem;
    margin-top: 2rem;
    padding: 1rem 0;
    border-top: 1px solid #e8eaf0;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8eaf0 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem !important;
}
.sidebar-header {
    background: #1a73e8;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.sidebar-header h3 {
    color: #ffffff;
    font-size: 1rem;
    font-weight: 700;
    margin: 0;
}
.sidebar-header p {
    color: #c8e0ff;
    font-size: 0.8rem;
    margin: 0.2rem 0 0 0;
}
</style>
""", unsafe_allow_html=True)

# ---- Train Model ----
@st.cache_resource
def train_model():
    df = pd.read_csv('Advertising.csv', index_col=0)
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    return model, df, metrics

model, df, metrics = train_model()

# ---- Sidebar ----
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <h3>📈 Sales Predictor</h3>
            <p>Adjust budgets to predict</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("**📺 TV Budget**")
    tv = st.slider("TV", min_value=0.0, max_value=300.0,
                   value=150.0, step=0.5, label_visibility="collapsed")

    st.markdown("**📻 Radio Budget**")
    radio = st.slider("Radio", min_value=0.0, max_value=50.0,
                      value=25.0, step=0.5, label_visibility="collapsed")

    st.markdown("**📰 Newspaper Budget**")
    newspaper = st.slider("Newspaper", min_value=0.0, max_value=120.0,
                          value=30.0, step=0.5, label_visibility="collapsed")

    st.divider()

    input_data = pd.DataFrame(
        {'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
    prediction = model.predict(input_data)[0]

    st.markdown(f"""
        <div class="result-display">
            <p class="result-label">Predicted Sales</p>
            <p class="result-value">{prediction:.2f}K</p>
            <p class="result-sub">thousand units</p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
        <div class="budget-row">
            <span class="budget-label">📺 TV</span>
            <span class="budget-value">${tv:.1f}K</span>
        </div>
        <div class="budget-row">
            <span class="budget-label">📻 Radio</span>
            <span class="budget-value">${radio:.1f}K</span>
        </div>
        <div class="budget-row">
            <span class="budget-label">📰 Newspaper</span>
            <span class="budget-value">${newspaper:.1f}K</span>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
        <div style='text-align:center; color:#aaaaaa; font-size:0.8rem; line-height:1.6'>
            <strong style='color:#555'>Vinay Goud</strong><br>
            3rd Year CSE (Data Science)<br>
            MRCET | Oasis Infobyte
        </div>
    """, unsafe_allow_html=True)

# ---- Main Area ----

# Header
st.markdown("""
    <div class="header-banner">
        <div>
            <h1>📈 Sales Prediction System</h1>
            <p>Predict product sales from TV, Radio & Newspaper advertising budgets using Machine Learning</p>
        </div>
        <div class="header-badge">R² Score: 0.90 ✅</div>
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
            <p class="metric-value">{metrics['mae']:.2f}</p>
            <p class="metric-label">Mean Abs Error</p>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <p class="metric-value">{metrics['rmse']:.2f}</p>
            <p class="metric-label">RMSE Score</p>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🗃️</div>
            <p class="metric-value">200</p>
            <p class="metric-label">Dataset Size</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Charts
st.markdown('<p class="section-title">📊 Data Visualizations</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
    fig1.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')
    corr_values = [
        df['TV'].corr(df['Sales']),
        df['Radio'].corr(df['Sales']),
        df['Newspaper'].corr(df['Sales'])
    ]
    bars = ax1.bar(['TV', 'Radio', 'Newspaper'], corr_values,
                   color=['#1a73e8', '#34a853', '#ea4335'],
                   width=0.5, edgecolor='none', zorder=3)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Correlation Score', fontsize=10, color='#555555')
    ax1.set_title('Ad Channel Correlation with Sales',
                  fontsize=11, fontweight='bold', color='#1a1a2e', pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.tick_params(colors='#888888')
    for bar, val in zip(bars, corr_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color='#1a1a2e')
    plt.tight_layout()
    st.pyplot(fig1)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    fig2.patch.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')
    ax2.scatter(df['TV'], df['Sales'],
                color='#1a73e8', alpha=0.5, s=35, zorder=3)
    m, b = np.polyfit(df['TV'], df['Sales'], 1)
    x_line = np.linspace(df['TV'].min(), df['TV'].max(), 100)
    ax2.plot(x_line, m * x_line + b,
             color='#ea4335', linewidth=2, zorder=4)
    ax2.set_xlabel('TV Budget ($thousands)', fontsize=10, color='#555555')
    ax2.set_ylabel('Sales (thousands)', fontsize=10, color='#555555')
    ax2.set_title('TV Budget vs Sales',
                  fontsize=11, fontweight='bold', color='#1a1a2e', pad=12)
    ax2.grid(linestyle='--', alpha=0.4, zorder=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(colors='#888888')
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

# Insights
st.markdown('<p class="section-title">💡 Key Business Insights</p>',
            unsafe_allow_html=True)
st.markdown("""
    <div class="insight-grid">
        <div class="insight-card">
            <div class="insight-icon">📺</div>
            <h4>TV drives volume</h4>
            <p>TV advertising has the strongest overall impact on sales with 0.78 correlation. Higher TV spend consistently delivers higher revenue.</p>
        </div>
        <div class="insight-card radio">
            <div class="insight-icon">📻</div>
            <h4>Radio = best ROI</h4>
            <p>Radio delivers the highest return per dollar spent with 0.576 correlation. Ideal channel for budget-conscious campaigns.</p>
        </div>
        <div class="insight-card newspaper">
            <div class="insight-icon">📰</div>
            <h4>Newspaper underperforms</h4>
            <p>Newspaper shows minimal sales impact with only 0.228 correlation. Consider reallocating this budget to TV or Radio.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Built by <strong>Vinay Goud</strong> &nbsp;|&nbsp;
        3rd Year CSE (Data Science) — MRCET &nbsp;|&nbsp;
        Oasis Infobyte Data Science Internship &nbsp;|&nbsp;
        Task 5 — Sales Prediction using Python
    </div>
""", unsafe_allow_html=True)