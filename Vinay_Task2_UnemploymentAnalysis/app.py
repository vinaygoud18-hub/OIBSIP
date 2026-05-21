# Unemployment Analysis Dashboard
# Oasis Infobyte Data Science Internship — Task 2
# Author: Vinay Goud | 3rd Year CSE (Data Science) — MRCET

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Unemployment Analysis | Vinay Goud",
    page_icon="📊",
    layout="wide"
)

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
    background: #1a73e8;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
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
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 1.5rem 0 1rem 0;
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
.chart-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #e8eaf0;
    margin-bottom: 1rem;
}
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
.insight-card.second { border-top-color: #ea4335; }
.insight-card.third  { border-top-color: #34a853; }
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

# ---- Load & Clean Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("Unemployment.csv")
    df.columns = df.columns.str.strip()
    df.rename(columns={
        'Region': 'State',
        'Estimated Unemployment Rate (%)': 'Unemployment_Rate',
        'Estimated Employed': 'Employed',
        'Estimated Labour Participation Rate (%)': 'Labour_Participation_Rate',
        'Area': 'Area'
    }, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Month_Name'] = df['Date'].dt.strftime('%b %Y')
    return df

df = load_data()

# ---- Sidebar Filters ----
with st.sidebar:
    st.markdown("""
        <div style='background:#1a73e8; border-radius:12px;
        padding:1rem; text-align:center; margin-bottom:1.5rem;'>
            <h3 style='color:#fff; margin:0; font-size:1rem;'>📊 Filters</h3>
            <p style='color:#c8e0ff; margin:0.2rem 0 0; font-size:0.8rem;'>
            Customize your analysis</p>
        </div>
    """, unsafe_allow_html=True)

    # Area filter
    area_options = ['All'] + list(df['Area'].unique())
    selected_area = st.selectbox("📍 Area Type", area_options)

    # State filter
    state_options = ['All'] + sorted(df['State'].dropna().astype(str).unique().tolist())
    selected_state = st.selectbox("🗺️ Select State", state_options)

    st.divider()
    st.markdown("""
        <div style='text-align:center; color:#aaa; font-size:0.8rem; line-height:1.6'>
            <strong style='color:#555'>Vinay Goud</strong><br>
            3rd Year CSE (Data Science)<br>
            MRCET | Oasis Infobyte<br>
            Task 2
        </div>
    """, unsafe_allow_html=True)

# ---- Apply Filters ----
filtered_df = df.copy()
if selected_area != 'All':
    filtered_df = filtered_df[filtered_df['Area'] == selected_area]
if selected_state != 'All':
    filtered_df = filtered_df[filtered_df['State'] == selected_state]

# ---- Header ----
st.markdown("""
    <div class="header-banner">
        <h1>📊 Unemployment Analysis — India</h1>
        <p>Analyzing unemployment trends across Indian states
        during COVID-19 (2020) using Data Science</p>
    </div>
""", unsafe_allow_html=True)

# ---- Metric Cards ----
avg_unemployment = filtered_df['Unemployment_Rate'].mean()
max_unemployment = filtered_df['Unemployment_Rate'].max()
min_unemployment = filtered_df['Unemployment_Rate'].min()
total_states = filtered_df['State'].nunique()

st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-icon">📉</div>
            <p class="metric-value">{avg_unemployment:.1f}%</p>
            <p class="metric-label">Avg Unemployment</p>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🔴</div>
            <p class="metric-value">{max_unemployment:.1f}%</p>
            <p class="metric-label">Peak Rate</p>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🟢</div>
            <p class="metric-value">{min_unemployment:.1f}%</p>
            <p class="metric-label">Lowest Rate</p>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🗺️</div>
            <p class="metric-value">{total_states}</p>
            <p class="metric-label">States Analyzed</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---- Chart 1: Monthly Trend ----
st.markdown('<p class="section-title">📈 Unemployment Rate Trend Over Time</p>',
            unsafe_allow_html=True)
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
monthly = filtered_df.groupby('Date')['Unemployment_Rate'].mean().reset_index()
fig1 = px.line(monthly, x='Date', y='Unemployment_Rate',
               markers=True,
               labels={'Unemployment_Rate': 'Unemployment Rate (%)',
                       'Date': 'Month'},
               color_discrete_sequence=['#1a73e8'])
fig1.update_layout(
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(color='#1a1a2e'),
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode='x unified'
)
fig1.update_traces(fill='tozeroy', fillcolor='rgba(26,115,232,0.08)')
st.plotly_chart(fig1, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---- Chart 2 & 3 ----
st.markdown('<p class="section-title">📊 State & Area Analysis</p>',
            unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    state_avg = filtered_df.groupby('State')['Unemployment_Rate']\
        .mean().sort_values(ascending=False).head(10).reset_index()
    fig2 = px.bar(state_avg, x='Unemployment_Rate', y='State',
                  orientation='h',
                  labels={'Unemployment_Rate': 'Avg Unemployment Rate (%)'},
                  color='Unemployment_Rate',
                  color_continuous_scale='Reds')
    fig2.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#1a1a2e'),
        margin=dict(l=20, r=20, t=30, b=20),
        title=dict(text='Top 10 States by Unemployment',
                   font=dict(size=13, color='#1a1a2e')),
        coloraxis_showscale=False,
        yaxis=dict(autorange='reversed')
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    area_avg = filtered_df.groupby('Area')['Unemployment_Rate']\
        .mean().reset_index()
    fig3 = px.pie(area_avg, values='Unemployment_Rate', names='Area',
                  color_discrete_sequence=['#1a73e8', '#34a853'],
                  hole=0.4)
    fig3.update_layout(
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#1a1a2e'),
        margin=dict(l=20, r=20, t=30, b=20),
        title=dict(text='Urban vs Rural Unemployment',
                   font=dict(size=13, color='#1a1a2e')),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2)
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Chart 4: Heatmap ----
st.markdown('<p class="section-title">🗺️ State-wise Monthly Heatmap</p>',
            unsafe_allow_html=True)
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
pivot = filtered_df.pivot_table(
    values='Unemployment_Rate',
    index='State',
    columns='Month',
    aggfunc='mean'
)
fig4 = px.imshow(pivot,
                 color_continuous_scale='YlOrRd',
                 labels=dict(x='Month', y='State',
                             color='Unemployment Rate (%)'),
                 aspect='auto')
fig4.update_layout(
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font=dict(color='#1a1a2e'),
    margin=dict(l=20, r=20, t=20, b=20)
)
st.plotly_chart(fig4, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---- Insights ----
st.markdown('<p class="section-title">💡 Key Insights</p>',
            unsafe_allow_html=True)
st.markdown("""
    <div class="insight-grid">
        <div class="insight-card">
            <div style='font-size:1.5rem; margin-bottom:0.5rem'>📈</div>
            <h4>COVID-19 Impact</h4>
            <p>Unemployment spiked sharply in April–May 2020
            during the national lockdown, reaching record
            highs across all Indian states.</p>
        </div>
        <div class="insight-card second">
            <div style='font-size:1.5rem; margin-bottom:0.5rem'>🗺️</div>
            <h4>Regional Disparity</h4>
            <p>States like Tripura, Haryana and Jharkhand
            recorded the highest unemployment rates while
            southern states showed more resilience.</p>
        </div>
        <div class="insight-card third">
            <div style='font-size:1.5rem; margin-bottom:0.5rem'>🏙️</div>
            <h4>Urban vs Rural</h4>
            <p>Urban areas experienced higher unemployment
            than rural areas, as city-based industries
            were more affected by lockdown restrictions.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---- Footer ----
st.markdown("""
    <div class="footer">
        Built by <strong>Vinay Goud</strong> &nbsp;|&nbsp;
        3rd Year CSE (Data Science) — MRCET &nbsp;|&nbsp;
        Oasis Infobyte Data Science Internship &nbsp;|&nbsp;
        Task 2 — Unemployment Analysis with Python
    </div>
""", unsafe_allow_html=True)