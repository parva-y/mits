#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# MITS - Marketing Intelligence & Testing Suite (Enhanced Version with Fixes)


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta
import random
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import traceback

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MITS - Marketing Intelligence & Testing Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .demo-banner {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }

    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }

    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }

    .insight-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #00acc1;
    }

    .recommendation-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ff6f00;
    }

    .analysis-step {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #6c757d;
    }

    .causal-card {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #8e24aa;
    }

    .error-card {
        background: #fee;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #f44336;
        color: #d32f2f;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    .ai-chat-container {
        background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'demo_step' not in st.session_state:
    st.session_state.demo_step = 0
if 'rca_results' not in st.session_state:
    st.session_state.rca_results = {}
if 'ai_messages' not in st.session_state:
    st.session_state.ai_messages = []

# Error handling decorator
def handle_errors(func):
    """Decorator to handle errors gracefully"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.markdown(f"""
            <div class="error-card">
                <h4>⚠️ An error occurred</h4>
                <p>We encountered an issue while processing your request. Please try again or contact support if the problem persists.</p>
                <details>
                    <summary>Technical details</summary>
                    <code>{str(e)}</code>
                </details>
            </div>
            """, unsafe_allow_html=True)
            return None
    return wrapper

# Data validation function
def validate_dataframe(df):
    """Validate and fix common dataframe issues"""
    try:
        # Ensure date column exists and is datetime
        date_columns = ['date', 'Date', 'DATE', 'created_at', 'timestamp']
        date_col_found = False

        for col in date_columns:
            if col in df.columns:
                try:
                    df['date'] = pd.to_datetime(df[col])
                    date_col_found = True
                    break
                except:
                    pass

        if not date_col_found and len(df) > 0:
            # Create a date column if none exists
            start_date = pd.Timestamp('2024-01-01')
            end_date = pd.Timestamp('2024-12-31')
            df['date'] = pd.date_range(start=start_date, periods=len(df), freq='D')[:len(df)]

        # Ensure numeric columns are numeric
        numeric_columns = ['impressions', 'clicks', 'conversions', 'cost', 'revenue', 
                          'ctr', 'conversion_rate', 'roas', 'cpm', 'cpc']

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Calculate missing metrics
        if 'impressions' in df.columns and 'clicks' in df.columns:
            if 'ctr' not in df.columns:
                df['ctr'] = np.where(df['impressions'] > 0, 
                                    df['clicks'] / df['impressions'], 0)

        if 'clicks' in df.columns and 'conversions' in df.columns:
            if 'conversion_rate' not in df.columns:
                df['conversion_rate'] = np.where(df['clicks'] > 0, 
                                                df['conversions'] / df['clicks'], 0)

        if 'revenue' in df.columns and 'cost' in df.columns:
            if 'roas' not in df.columns:
                df['roas'] = np.where(df['cost'] > 0, 
                                     df['revenue'] / df['cost'], 0)

        # Add default columns if missing
        if 'campaign_name' not in df.columns:
            df['campaign_name'] = 'Default_Campaign'

        if 'channel' not in df.columns:
            df['channel'] = 'Unknown'

        return df, True

    except Exception as e:
        return df, False

# Enhanced data generation function with demographic data
@st.cache_data
@handle_errors
def generate_demo_data():
    """Generate realistic marketing demo data with patterns for RCA"""
    np.random.seed(42)

    # Date range
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2024-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Campaign configurations
    campaigns = ['Summer_Sale_2024', 'Holiday_Promo_2024', 'Flash_Deal_2024', 
                'Premium_Launch_2024', 'Spring_Campaign_2024']
    channels = ['Email', 'Social Media', 'Display', 'Search', 'Video']
    sources = ['google', 'facebook', 'instagram', 'linkedin', 'direct', 'organic', 'newsletter']
    mediums = ['cpc', 'cpm', 'organic', 'email', 'social', 'referral']
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    countries = ['United States', 'United Kingdom', 'Germany', 'Canada', 'Australia', 'France', 'Japan', 'Brazil']
    cities = ['New York', 'London', 'Berlin', 'Toronto', 'Sydney', 'Paris', 'Tokyo', 'São Paulo']
    devices = ['Desktop', 'Mobile', 'Tablet']
    segments = ['Premium', 'Regular', 'Budget', 'Enterprise', 'SMB']
    genders = ['Male', 'Female', 'Other']
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']

    # Channel performance characteristics
    channel_config = {
        'Email': {'base_ctr': 0.025, 'base_conv': 0.02, 'base_cpm': 5},
        'Social Media': {'base_ctr': 0.015, 'base_conv': 0.015, 'base_cpm': 8},
        'Display': {'base_ctr': 0.001, 'base_conv': 0.01, 'base_cpm': 3},
        'Search': {'base_ctr': 0.03, 'base_conv': 0.03, 'base_cpm': 15},
        'Video': {'base_ctr': 0.02, 'base_conv': 0.025, 'base_cpm': 12}
    }

    # Demographics performance modifiers
    age_modifiers = {
        '18-24': {'conv_mult': 0.8, 'revenue_mult': 0.7},
        '25-34': {'conv_mult': 1.2, 'revenue_mult': 1.0},
        '35-44': {'conv_mult': 1.1, 'revenue_mult': 1.3},
        '45-54': {'conv_mult': 0.9, 'revenue_mult': 1.5},
        '55-64': {'conv_mult': 0.7, 'revenue_mult': 1.8},
        '65+': {'conv_mult': 0.6, 'revenue_mult': 2.0}
    }

    data = []

    for date in dates:
        # Seasonal factor
        day_of_year = date.dayofyear
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)

        # Weekend factor
        weekend_factor = 0.7 if date.dayofweek in [5, 6] else 1.0

        # Holiday boost
        holiday_boost = 1.0
        if date.month == 11 and date.day >= 24 and date.day <= 30:  # Black Friday week
            holiday_boost = 3.0
        elif date.month == 12 and date.day <= 25:  # December shopping
            holiday_boost = 1.8

        # Create artificial events for root cause analysis
        # Event 1: iOS update impact on June 15
        ios_impact = 1.0
        if date >= pd.Timestamp('2024-06-15') and date <= pd.Timestamp('2024-07-15'):
            ios_impact = 0.6  # 40% drop in iOS performance

        # Event 2: Competitor launch impact
        competitor_impact = 1.0
        if date >= pd.Timestamp('2024-09-01') and date <= pd.Timestamp('2024-09-30'):
            competitor_impact = 0.8  # 20% drop due to competitor

        # Generate 10-30 rows per day
        for _ in range(random.randint(10, 30)):
            campaign = random.choice(campaigns)
            channel = random.choice(channels)
            source = random.choice(sources)
            medium = random.choice(mediums)
            region = random.choice(regions)
            country = random.choice(countries)
            city = random.choice(cities)
            device = random.choice(devices)
            segment = random.choice(segments)
            gender = random.choice(genders)
            age_group = random.choice(age_groups)

            # Source/medium combination logic
            if source == 'google':
                medium = random.choice(['cpc', 'organic', 'cpm'])
            elif source == 'facebook' or source == 'instagram':
                medium = 'social'
            elif source == 'newsletter':
                medium = 'email'
            elif source == 'direct':
                medium = 'none'

            # Base metrics
            base_config = channel_config[channel]

            # Campaign performance multiplier
            campaign_mult = {
                'Summer_Sale_2024': 1.2,
                'Holiday_Promo_2024': 1.5,
                'Flash_Deal_2024': 0.9,
                'Premium_Launch_2024': 1.3,
                'Spring_Campaign_2024': 1.0
            }[campaign]

            # Apply iOS impact to Mobile devices
            device_impact = ios_impact if device == 'Mobile' else 1.0

            # Apply demographic modifiers
            age_mod = age_modifiers[age_group]

            # Generate metrics
            impressions = int(np.random.lognormal(8, 1.5) * seasonal_factor * 
                            weekend_factor * holiday_boost)

            ctr = np.clip(base_config['base_ctr'] * campaign_mult * device_impact *
                         competitor_impact * (1 + np.random.normal(0, 0.2)), 0.0001, 0.1)
            clicks = int(impressions * ctr)

            conv_rate = np.clip(base_config['base_conv'] * campaign_mult * device_impact *
                              competitor_impact * age_mod['conv_mult'] * (1 + np.random.normal(0, 0.15)), 0.0001, 0.2)
            conversions = int(clicks * conv_rate)

            cpm = base_config['base_cpm'] * (1 + np.random.normal(0, 0.2))
            cost = (impressions / 1000) * cpm

            # Revenue based on segment and age
            revenue_per_conv = {
                'Premium': 150, 'Enterprise': 200, 'Regular': 50, 
                'Budget': 20, 'SMB': 80
            }[segment] * age_mod['revenue_mult'] * (1 + np.random.normal(0, 0.3))
            revenue = conversions * revenue_per_conv

            # Additional metrics
            bounce_rate = np.clip(0.4 + np.random.normal(0, 0.1), 0.1, 0.9)
            quality_score = np.clip(5 + campaign_mult * 2 + np.random.normal(0, 1), 1, 10)

            # Special patterns
            # Make Display + Latin America underperform
            if channel == 'Display' and region == 'Latin America':
                conversions = int(conversions * 0.3)
                revenue = revenue * 0.3
                ctr = ctr * 0.5

            # Make Search perform better
            if channel == 'Search':
                conversions = int(conversions * 1.5)
                revenue = revenue * 1.5

            # Add technical issue on specific date
            if date == pd.Timestamp('2024-06-15'):
                conversions = int(conversions * 0.1)
                clicks = int(clicks * 0.2)
                revenue = revenue * 0.1

            record = {
                'date': date,
                'campaign_name': campaign,
                'channel': channel,
                'source': source,
                'medium': medium,
                'source_medium': f"{source} / {medium}",
                'region': region,
                'country': country,
                'city': city,
                'device_type': device,
                'customer_segment': segment,
                'gender': gender,
                'age_group': age_group,
                'impressions': impressions,
                'clicks': clicks,
                'conversions': conversions,
                'cost': round(cost, 2),
                'revenue': round(revenue, 2),
                'ctr': round(ctr, 4),
                'conversion_rate': round(conv_rate, 4),
                'cpm': round(cpm, 2),
                'cpc': round(cost / clicks if clicks > 0 else 0, 2),
                'roas': round(revenue / cost if cost > 0 else 0, 2),
                'bounce_rate': round(bounce_rate, 3),
                'quality_score': round(quality_score, 1),
                'day_of_week': date.strftime('%A'),
                'month': date.strftime('%B'),
                'quarter': f'Q{(date.month-1)//3 + 1}'
            }

            data.append(record)

    df = pd.DataFrame(data)
    return validate_dataframe(df)[0]

# Root Cause Analysis Functions with error handling
@handle_errors
def detect_anomalies_advanced(df, kpi, method='isolation_forest'):
    """Advanced anomaly detection using multiple methods"""
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN

    # Prepare data
    daily_data = df.groupby('date')[kpi].agg(['mean', 'sum', 'count']).reset_index()

    if method == 'isolation_forest':
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        daily_data['anomaly'] = iso_forest.fit_predict(daily_data[['mean', 'sum']])
        daily_data['anomaly'] = daily_data['anomaly'].map({1: False, -1: True})

    elif method == 'statistical':
        # Statistical method (Z-score)
        daily_data['z_score'] = np.abs(stats.zscore(daily_data['mean']))
        daily_data['anomaly'] = daily_data['z_score'] > 2.5

    return daily_data

@handle_errors
def correlation_analysis(df, target_kpi):
    """Perform correlation analysis to find related factors"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_kpi]

    correlations = []
    for col in numeric_cols:
        if df[target_kpi].std() > 0 and df[col].std() > 0:
            corr, p_value = stats.pearsonr(df[target_kpi].fillna(0), df[col].fillna(0))
            correlations.append({
                'feature': col,
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

    return pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)

@handle_errors
def causal_inference_its(df, intervention_date, kpi):
    """Interrupted Time Series analysis"""
    # Prepare data
    daily_data = df.groupby('date')[kpi].mean().reset_index()
    daily_data['time'] = range(len(daily_data))
    daily_data['intervention'] = (daily_data['date'] >= intervention_date).astype(int)
    daily_data['time_after'] = daily_data['time'] * daily_data['intervention']

    # Fit regression model
    X = daily_data[['time', 'intervention', 'time_after']]
    y = daily_data[kpi]

    model = sm.OLS(y, sm.add_constant(X)).fit()

    return model, daily_data

@handle_errors
def difference_in_differences(df, treatment_group, control_group, treatment_date, kpi, group_col):
    """Difference-in-Differences analysis"""
    # Filter data
    did_data = df[df[group_col].isin([treatment_group, control_group])].copy()
    did_data['treatment'] = (did_data[group_col] == treatment_group).astype(int)
    did_data['post'] = (did_data['date'] >= treatment_date).astype(int)
    did_data['did'] = did_data['treatment'] * did_data['post']

    # Aggregate by date and group
    agg_data = did_data.groupby(['date', group_col])[kpi].mean().reset_index()

    # Fit DiD model
    X = did_data[['treatment', 'post', 'did']]
    y = did_data[kpi]

    model = sm.OLS(y, sm.add_constant(X)).fit()

    return model, agg_data

@handle_errors
def feature_importance_analysis(df, target_kpi):
    """Random Forest feature importance - excluding the target variable"""
    # Prepare data
    numeric_df = df.select_dtypes(include=[np.number]).dropna()

    # Encode categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()

    for col in cat_cols:
        if col in ['date', 'day_of_week', 'month']:
            continue
        if col in df.columns:
            df[f'{col}_encoded'] = le.fit_transform(df[col].fillna('missing'))
            if f'{col}_encoded' in df.columns:
                numeric_df[f'{col}_encoded'] = df[f'{col}_encoded']

    # Separate features and target - EXCLUDE target from features
    feature_cols = [col for col in numeric_df.columns if col != target_kpi and col in numeric_df.columns]

    # Also exclude derivatives of the target (e.g., if target is conversions, exclude conversion_rate)
    if target_kpi == 'conversions':
        feature_cols = [col for col in feature_cols if 'conversion' not in col.lower()]
    elif target_kpi == 'revenue':
        feature_cols = [col for col in feature_cols if 'revenue' not in col.lower() and 'roas' not in col.lower()]

    if len(feature_cols) == 0:
        return None, pd.DataFrame()

    X = numeric_df[feature_cols]
    y = numeric_df[target_kpi]

    # Fit Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return rf, importance_df

# Demo guide steps
demo_steps = [
    "Welcome! Let's explore the Executive Dashboard to see your marketing performance at a glance.",
    "Now check out Campaign Analysis - see how different campaigns compare and identify top performers.",
    "Let's use Root Cause Analysis to automatically detect anomalies in your KPIs.",
    "Time for Statistical Testing - validate your hypotheses with confidence.",
    "Finally, get AI-powered recommendations for optimizing your marketing strategy."
]

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 MITS – Marketing Intelligence & Testing Suite</h1>
    <p>Transform Your Marketing Data into Actionable Insights</p>
</div>
""", unsafe_allow_html=True)

# Demo mode controls
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🎯 Start Product Tour" if not st.session_state.demo_mode else "📊 Exit Product Tour", 
                 use_container_width=True):
        st.session_state.demo_mode = not st.session_state.demo_mode
        st.session_state.demo_step = 0
        if st.session_state.demo_mode:
            with st.spinner("🎨 Creating perfect demo data..."):
                demo_data = generate_demo_data()
                if demo_data is not None:
                    st.session_state.data = demo_data
                else:
                    st.error("Failed to generate demo data. Please try again.")
                    st.session_state.demo_mode = False
            st.rerun()

# Demo mode banner and navigation
if st.session_state.demo_mode:
    st.markdown(f"""
    <div class="demo-banner">
        🎯 Product Tour - Step {st.session_state.demo_step + 1}/{len(demo_steps)}: {demo_steps[st.session_state.demo_step]}
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Previous") and st.session_state.demo_step > 0:
            st.session_state.demo_step -= 1
            st.rerun()
    with col3:
        if st.button("Next ➡️") and st.session_state.demo_step < len(demo_steps) - 1:
            st.session_state.demo_step += 1
            st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Data Management")

    data_source = st.radio(
        "Select Data Source",
        ["Use Demo Data", "Upload File"],
        index=0 if st.session_state.demo_mode else 1
    )

    if data_source == "Upload File":
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # Validate and clean data
                df, valid = validate_dataframe(df)

                if valid:
                    st.session_state.data = df
                    st.success(f"✅ Loaded {len(df):,} rows")
                else:
                    st.warning("Data loaded with some issues. We've attempted to fix them.")
                    st.session_state.data = df

            except Exception as e:
                st.error(f"Error loading file. Using demo data instead.")
                st.info("Please ensure your file is properly formatted.")
                if st.button("Load Demo Data Instead"):
                    st.session_state.data = generate_demo_data()
                    st.rerun()

    else:
        if st.button("🎲 Generate Fresh Demo Data"):
            with st.spinner("Creating data..."):
                demo_data = generate_demo_data()
                if demo_data is not None:
                    st.session_state.data = demo_data
                    st.success("✅ Demo data ready!")
                    st.rerun()

    # Data overview
    if st.session_state.data is not None:
        st.markdown("---")
        st.markdown("### 📈 Data Overview")
        df = st.session_state.data

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
            if 'campaign_name' in df.columns:
                st.metric("Campaigns", df['campaign_name'].nunique())
        with col2:
            if 'date' in df.columns:
                try:
                    st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
                except:
                    st.metric("Date Range", "Invalid dates")
            if 'channel' in df.columns:
                st.metric("Channels", df['channel'].nunique())

# Main content
if st.session_state.data is not None:
    df = st.session_state.data

    # Create tabs
    tab_list = ["📊 Executive Dashboard", "📈 Campaign Analysis", "🔍 Root Cause Analysis", 
                "🧪 Statistical Testing", "🤖 AI Recommendations"]

    # Highlight current tab in demo mode
    if st.session_state.demo_mode:
        tab_list[st.session_state.demo_step] = f"👉 {tab_list[st.session_state.demo_step]}"

    tabs = st.tabs(tab_list)

    # Tab 1: Executive Dashboard
    with tabs[0]:
        st.markdown("## Executive Dashboard")

        # Calculate KPIs with error handling
        try:
            total_revenue = df['revenue'].sum() if 'revenue' in df.columns else 0
            total_cost = df['cost'].sum() if 'cost' in df.columns else 0
            total_conversions = df['conversions'].sum() if 'conversions' in df.columns else 0
            avg_roas = df['roas'].mean() if 'roas' in df.columns else 0

            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <h3 style="color: #667eea;">Total Revenue</h3>
                    <h1>${total_revenue:,.0f}</h1>
                    <p style="color: green;">↑ 23.5% vs last period</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <h3 style="color: #764ba2;">Total Conversions</h3>
                    <h1>{total_conversions:,.0f}</h1>
                    <p style="color: green;">↑ 15.2% vs last period</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="kpi-card">
                    <h3 style="color: #9f7aea;">Average ROAS</h3>
                    <h1>{avg_roas:.2f}x</h1>
                    <p style="color: green;">↑ 0.3 vs last period</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                profit_margin = ((total_revenue - total_cost) / total_revenue * 100) if total_revenue > 0 else 0
                st.markdown(f"""
                <div class="kpi-card">
                    <h3 style="color: #b794f4;">Profit Margin</h3>
                    <h1>{profit_margin:.1f}%</h1>
                    <p style="color: green;">↑ 2.1% vs last period</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Visualizations with error handling
            col1, col2 = st.columns([2, 1])

            with col1:
                try:
                    if 'date' in df.columns and 'revenue' in df.columns:
                        # Revenue trend
                        daily_revenue = df.groupby('date')['revenue'].sum().reset_index()

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=daily_revenue['date'],
                            y=daily_revenue['revenue'],
                            mode='lines',
                            name='Daily Revenue',
                            line=dict(color='#667eea', width=3),
                            fill='tonexty',
                            fillcolor='rgba(102, 126, 234, 0.1)'
                        ))

                        # Add 7-day moving average
                        daily_revenue['ma7'] = daily_revenue['revenue'].rolling(7, min_periods=1).mean()
                        fig.add_trace(go.Scatter(
                            x=daily_revenue['date'],
                            y=daily_revenue['ma7'],
                            mode='lines',
                            name='7-day Average',
                            line=dict(color='#764ba2', width=2, dash='dash')
                        ))

                        fig.update_layout(
                            title="Revenue Trend Over Time",
                            xaxis_title="Date",
                            yaxis_title="Revenue ($)",
                            hovermode='x unified',
                            height=400,
                            showlegend=True
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Date and revenue columns needed for trend visualization")
                except Exception as e:
                    st.error("Unable to create revenue trend chart. Please check your data.")

            with col2:
                try:
                    if 'channel' in df.columns and 'revenue' in df.columns:
                        # Channel pie chart
                        channel_revenue = df.groupby('channel')['revenue'].sum()

                        fig = px.pie(
                            values=channel_revenue.values,
                            names=channel_revenue.index,
                            title="Revenue by Channel",
                            color_discrete_sequence=px.colors.sequential.Plasma
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Channel data needed for this visualization")
                except Exception as e:
                    st.error("Unable to create channel pie chart.")

            # Performance heatmap with error handling
            try:
                if all(col in df.columns for col in ['channel', 'campaign_name', 'roas']):
                    st.markdown("### Performance Matrix")

                    # Create pivot table
                    pivot_data = df.groupby(['channel', 'campaign_name'])['roas'].mean().reset_index()
                    pivot_table = pivot_data.pivot(index='channel', columns='campaign_name', values='roas').fillna(0)

                    fig = px.imshow(
                        pivot_table,
                        labels=dict(x="Campaign", y="Channel", color="ROAS"),
                        color_continuous_scale='RdYlGn',
                        title="ROAS Heatmap by Channel and Campaign",
                        aspect='auto'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info("Unable to create performance matrix. Check if required columns exist.")

            # Additional charts with variable selection
            st.markdown("### 📊 Custom Analytics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Variable Comparison")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                x_var = st.selectbox("X-axis variable", numeric_cols, index=numeric_cols.index('cost') if 'cost' in numeric_cols else 0)
                y_var = st.selectbox("Y-axis variable", numeric_cols, index=numeric_cols.index('revenue') if 'revenue' in numeric_cols else 1)

                if st.button("Create Scatter Plot"):
                    fig = px.scatter(df, x=x_var, y=y_var, 
                                   color='channel' if 'channel' in df.columns else None,
                                   size='conversions' if 'conversions' in df.columns else None,
                                   title=f"{y_var} vs {x_var}",
                                   hover_data=['campaign_name'] if 'campaign_name' in df.columns else None)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Distribution Analysis")
                dist_var = st.selectbox("Select variable for distribution", numeric_cols)
                group_by = st.selectbox("Group by", ['None'] + [col for col in df.columns if col in ['channel', 'campaign_name', 'device_type', 'region']])

                if st.button("Create Distribution Plot"):
                    if group_by == 'None':
                        fig = px.histogram(df, x=dist_var, nbins=30, 
                                         title=f"Distribution of {dist_var}")
                    else:
                        fig = px.box(df, x=group_by, y=dist_var, 
                                    title=f"{dist_var} by {group_by}")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

            # Time Series Analysis - Campaign vs Baseline
            st.markdown("### 📈 Time Series Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                if 'date' in df.columns:
                    campaign_start = pd.to_datetime(
                        st.date_input(
                            "Campaign Start Date",
                            value=df['date'].min() + pd.Timedelta(days=30)
                        )
                    )
                    campaign_end = pd.to_datetime(
                        st.date_input(
                            "Campaign End Date",
                            value=df['date'].min() + pd.Timedelta(days=60)
                        )
                    )

            with col2:
                baseline_days = st.number_input("Baseline Period (days before campaign)", 
                                              min_value=7, max_value=90, value=30)

            with col3:
                ts_metric = st.selectbox("Metric for Time Series", 
                                       ['revenue', 'conversions', 'roas', 'ctr'])

            if st.button("Run Time Series Analysis", use_container_width=True):
                try:
                    # Define periods
                    baseline_start = pd.to_datetime(campaign_start) - pd.Timedelta(days=baseline_days)
                    baseline_end = pd.to_datetime(campaign_start) - pd.Timedelta(days=1)

                    # Filter data
                    baseline_data = df[(df['date'] >= baseline_start) & (df['date'] <= baseline_end)]
                    campaign_data = df[(df['date'] >= pd.to_datetime(campaign_start)) & 
                                     (df['date'] <= pd.to_datetime(campaign_end))]

                    # Aggregate by date
                    baseline_daily = baseline_data.groupby('date')[ts_metric].mean().reset_index()
                    campaign_daily = campaign_data.groupby('date')[ts_metric].mean().reset_index()

                    # Create comparison chart
                    fig = go.Figure()

                    # Baseline period
                    fig.add_trace(go.Scatter(
                        x=baseline_daily['date'],
                        y=baseline_daily[ts_metric],
                        mode='lines+markers',
                        name='Baseline Period',
                        line=dict(color='#667eea', width=2)
                    ))

                    # Campaign period
                    fig.add_trace(go.Scatter(
                        x=campaign_daily['date'],
                        y=campaign_daily[ts_metric],
                        mode='lines+markers',
                        name='Campaign Period',
                        line=dict(color='#764ba2', width=2)
                    ))

                    # Add vertical line for campaign start
                    fig.add_vline(x=pd.to_datetime(campaign_start), 
                                line_dash="dash", line_color="red",
                                annotation_text="Campaign Start")

                    # Calculate lift
                    baseline_avg = baseline_daily[ts_metric].mean()
                    campaign_avg = campaign_daily[ts_metric].mean()
                    lift = ((campaign_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0

                    fig.update_layout(
                        title=f"{ts_metric.upper()} - Campaign vs Baseline (Lift: {lift:+.1f}%)",
                        xaxis_title="Date",
                        yaxis_title=ts_metric,
                        height=400,
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Baseline Average", f"{baseline_avg:.2f}")
                    with col2:
                        st.metric("Campaign Average", f"{campaign_avg:.2f}")
                    with col3:
                        st.metric("Lift", f"{lift:+.1f}%", 
                                delta="Positive" if lift > 0 else "Negative")

                except Exception as e:
                    st.error("Error in time series analysis. Please check your date selections.")

        except Exception as e:
            st.error("Error creating dashboard. Please check your data format.")
            st.info("Ensure your data contains: revenue, cost, conversions, and date columns.")

    # Tab 2: Campaign Analysis
    with tabs[1]:
        st.markdown("## Campaign Performance Analysis")

        try:
            if 'campaign_name' in df.columns:
                campaigns = df['campaign_name'].unique()
                selected_campaigns = st.multiselect(
                    "Select Campaigns to Compare",
                    campaigns,
                    default=list(campaigns[:3]) if len(campaigns) >= 3 else list(campaigns)
                )

                if selected_campaigns:
                    campaign_df = df[df['campaign_name'].isin(selected_campaigns)]

                    col1, col2 = st.columns(2)

                    with col1:
                        # Revenue vs Cost comparison
                        if 'revenue' in df.columns and 'cost' in df.columns:
                            campaign_metrics = campaign_df.groupby('campaign_name').agg({
                                'revenue': 'sum',
                                'cost': 'sum'
                            }).reset_index()

                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                name='Revenue', 
                                x=campaign_metrics['campaign_name'], 
                                y=campaign_metrics['revenue'],
                                marker_color='#667eea'
                            ))
                            fig.add_trace(go.Bar(
                                name='Cost', 
                                x=campaign_metrics['campaign_name'], 
                                y=campaign_metrics['cost'],
                                marker_color='#764ba2'
                            ))

                            fig.update_layout(
                                title="Revenue vs Cost by Campaign",
                                barmode='group',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Conversion funnel
                        if all(col in df.columns for col in ['impressions', 'clicks', 'conversions']):
                            st.markdown("### Conversion Funnels")

                            for i, campaign in enumerate(selected_campaigns[:3]):  # Limit to 3
                                camp_data = campaign_df[campaign_df['campaign_name'] == campaign]

                                funnel_values = [
                                    camp_data['impressions'].sum(),
                                    camp_data['clicks'].sum(),
                                    camp_data['conversions'].sum()
                                ]

                                # Mini funnel chart
                                fig = go.Figure(go.Funnel(
                                    y=["Impressions", "Clicks", "Conversions"],
                                    x=funnel_values,
                                    textinfo="value+percent initial",
                                    marker={"color": ["#667eea", "#764ba2", "#9f7aea"]}
                                ))

                                fig.update_layout(
                                    title=f"{campaign[:20]}...",
                                    height=250,
                                    margin=dict(l=20, r=20, t=40, b=20)
                                )

                                st.plotly_chart(fig, use_container_width=True, key=f"funnel_{i}")

                    # Performance Analysis Section
                    st.markdown("### 📊 Performance Analysis")

                    # User Traffic Analysis
                    col1, col2 = st.columns(2)

                    with col1:
                        # Traffic by Source
                        if 'source' in df.columns:
                            st.markdown("#### Traffic % by Source")
                            source_traffic = campaign_df.groupby('source')['impressions'].sum()
                            source_pct = (source_traffic / source_traffic.sum() * 100).sort_values(ascending=False)

                            fig = px.bar(x=source_pct.values, y=source_pct.index, 
                                       orientation='h',
                                       title="Traffic Distribution by Source",
                                       labels={'x': 'Percentage (%)', 'y': 'Source'})
                            fig.update_traces(marker_color='#667eea')
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Traffic by Medium
                        if 'medium' in df.columns:
                            st.markdown("#### Traffic % by Medium")
                            medium_traffic = campaign_df.groupby('medium')['impressions'].sum()

                            fig = px.pie(values=medium_traffic.values, 
                                       names=medium_traffic.index,
                                       title="Traffic Distribution by Medium",
                                       color_discrete_sequence=px.colors.sequential.Plasma)
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

                    # Source/Medium Performance
                    if 'source_medium' in df.columns:
                        st.markdown("#### Source/Medium Performance Analysis")

                        # Calculate key metrics by source/medium
                        sm_metrics = campaign_df.groupby('source_medium').agg({
                            'impressions': 'sum',
                            'clicks': 'sum',
                            'conversions': 'sum',
                            'revenue': 'sum',
                            'cost': 'sum'
                        }).reset_index()

                        # Calculate derived metrics
                        sm_metrics['ctr'] = sm_metrics['clicks'] / sm_metrics['impressions'] * 100
                        sm_metrics['conversion_rate'] = sm_metrics['conversions'] / sm_metrics['clicks'] * 100
                        sm_metrics['roas'] = sm_metrics['revenue'] / sm_metrics['cost']
                        sm_metrics['profit'] = sm_metrics['revenue'] - sm_metrics['cost']

                        # Sort by profit
                        sm_metrics = sm_metrics.sort_values('profit', ascending=False)

                        # Display top performers
                        st.markdown("##### 🏆 Top Profit-Generating Sources")
                        top_sources = sm_metrics.head(10)

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=top_sources['source_medium'],
                            y=top_sources['profit'],
                            name='Profit',
                            marker_color='#667eea',
                            text=top_sources['profit'].apply(lambda x: f'${x:,.0f}'),
                            textposition='outside'
                        ))

                        fig.update_layout(
                            title="Top 10 Sources by Profit",
                            xaxis_title="Source / Medium",
                            yaxis_title="Profit ($)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Detailed metrics table
                        st.markdown("##### 📋 Detailed Source/Medium Metrics")
                        display_cols = ['source_medium', 'impressions', 'clicks', 'conversions', 
                                      'ctr', 'conversion_rate', 'revenue', 'cost', 'roas', 'profit']

                        styled_metrics = sm_metrics[display_cols].style.format({
                            'impressions': '{:,.0f}',
                            'clicks': '{:,.0f}',
                            'conversions': '{:,.0f}',
                            'ctr': '{:.2f}%',
                            'conversion_rate': '{:.2f}%',
                            'revenue': '${:,.0f}',
                            'cost': '${:,.0f}',
                            'roas': '{:.2f}x',
                            'profit': '${:,.0f}'
                        }).background_gradient(subset=['profit', 'roas'], cmap='RdYlGn')

                        st.dataframe(styled_metrics, use_container_width=True)

                    # Demographic Analysis
                    st.markdown("### 👥 Demographic Performance")

                    demo_cols = st.columns(2)

                    with demo_cols[0]:
                        # Gender Performance
                        if 'gender' in df.columns:
                            gender_metrics = campaign_df.groupby('gender').agg({
                                'conversions': 'sum',
                                'revenue': 'sum'
                            }).reset_index()

                            fig = make_subplots(rows=1, cols=2, 
                                              subplot_titles=('Conversions by Gender', 'Revenue by Gender'))

                            fig.add_trace(go.Bar(x=gender_metrics['gender'], 
                                               y=gender_metrics['conversions'],
                                               marker_color='#667eea'), row=1, col=1)

                            fig.add_trace(go.Bar(x=gender_metrics['gender'], 
                                               y=gender_metrics['revenue'],
                                               marker_color='#764ba2'), row=1, col=2)

                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)

                        # Age Group Performance
                        if 'age_group' in df.columns:
                            age_metrics = campaign_df.groupby('age_group').agg({
                                'conversions': 'sum',
                                'revenue': 'sum',
                                'clicks': 'sum'
                            }).reset_index()

                            age_metrics['conversion_rate'] = age_metrics['conversions'] / age_metrics['clicks'] * 100
                            age_metrics = age_metrics.sort_values('age_group')

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=age_metrics['age_group'],
                                y=age_metrics['conversion_rate'],
                                mode='lines+markers',
                                line=dict(color='#667eea', width=3),
                                marker=dict(size=10)
                            ))

                            fig.update_layout(
                                title="Conversion Rate by Age Group",
                                xaxis_title="Age Group",
                                yaxis_title="Conversion Rate (%)",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with demo_cols[1]:
                        # Device Performance
                        if 'device_type' in df.columns:
                            device_metrics = campaign_df.groupby('device_type').agg({
                                'impressions': 'sum',
                                'clicks': 'sum',
                                'conversions': 'sum',
                                'revenue': 'sum'
                            }).reset_index()

                            # Calculate percentages
                            total_revenue = device_metrics['revenue'].sum()
                            device_metrics['revenue_pct'] = device_metrics['revenue'] / total_revenue * 100

                            fig = px.pie(values=device_metrics['revenue_pct'], 
                                       names=device_metrics['device_type'],
                                       title="Revenue Distribution by Device",
                                       color_discrete_sequence=px.colors.sequential.Plasma)
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)

                        # Geographic Performance
                        if 'country' in df.columns:
                            country_revenue = campaign_df.groupby('country')['revenue'].sum().sort_values(ascending=False).head(10)

                            fig = px.bar(x=country_revenue.index, y=country_revenue.values,
                                       title="Top 10 Countries by Revenue",
                                       labels={'x': 'Country', 'y': 'Revenue ($)'},
                                       color=country_revenue.values,
                                       color_continuous_scale='Viridis')
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)

                    # Profit Analysis
                    st.markdown("### 💰 Profit Analysis - Pinpointing High-Value Segments")

                    # Multi-dimensional profit analysis
                    profit_dims = []
                    if 'source_medium' in df.columns:
                        profit_dims.append('source_medium')
                    if 'device_type' in df.columns:
                        profit_dims.append('device_type')
                    if 'age_group' in df.columns:
                        profit_dims.append('age_group')
                    if 'customer_segment' in df.columns:
                        profit_dims.append('customer_segment')

                    if len(profit_dims) >= 2:
                        dim1 = st.selectbox("Primary Dimension", profit_dims, index=0)
                        dim2 = st.selectbox("Secondary Dimension", 
                                          [d for d in profit_dims if d != dim1], index=0)

                        # Calculate profit by dimensions
                        profit_analysis = campaign_df.groupby([dim1, dim2]).agg({
                            'revenue': 'sum',
                            'cost': 'sum'
                        }).reset_index()
                        profit_analysis['profit'] = profit_analysis['revenue'] - profit_analysis['cost']

                        # Create pivot for heatmap
                        profit_pivot = profit_analysis.pivot(index=dim1, columns=dim2, values='profit').fillna(0)

                        fig = px.imshow(profit_pivot,
                                      labels=dict(x=dim2, y=dim1, color="Profit ($)"),
                                      title=f"Profit Heatmap: {dim1} vs {dim2}",
                                      color_continuous_scale='RdYlGn',
                                      aspect='auto')
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)

                        # Top profit segments
                        st.markdown("#### 🎯 Top Profit Segments")
                        top_segments = profit_analysis.nlargest(10, 'profit')

                        for idx, row in top_segments.iterrows():
                            profit_pct = row['profit'] / profit_analysis['profit'].sum() * 100
                            st.markdown(f"""
                            **{row[dim1]} × {row[dim2]}**  
                            Profit: ${row['profit']:,.0f} ({profit_pct:.1f}% of total)  
                            Revenue: ${row['revenue']:,.0f} | Cost: ${row['cost']:,.0f}
                            """)
                            if idx < 2:  # Show separator for top 3
                                st.markdown("---")

                    # Time Series Comparison
                    st.markdown("### 📈 Campaign Period Analysis")

                    if 'date' in df.columns:
                        selected_campaign = st.selectbox("Select Campaign for Time Analysis", selected_campaigns)

                        # Get campaign data
                        camp_specific = campaign_df[campaign_df['campaign_name'] == selected_campaign]

                        if len(camp_specific) > 0:
                            # Find campaign period
                            camp_start = camp_specific['date'].min()
                            camp_end = camp_specific['date'].max()

                            # Define baseline (30 days before campaign)
                            baseline_start = camp_start - pd.Timedelta(days=30)
                            baseline_end = camp_start - pd.Timedelta(days=1)

                            # Get all data for comparison
                            baseline_all = df[(df['date'] >= baseline_start) & (df['date'] <= baseline_end)]

                            # Aggregate metrics
                            baseline_daily = baseline_all.groupby('date').agg({
                                'revenue': 'sum',
                                'conversions': 'sum',
                                'clicks': 'sum',
                                'impressions': 'sum'
                            }).reset_index()

                            campaign_daily = camp_specific.groupby('date').agg({
                                'revenue': 'sum',
                                'conversions': 'sum',
                                'clicks': 'sum',
                                'impressions': 'sum'
                            }).reset_index()

                            # Create comparison charts
                            metrics_to_plot = ['revenue', 'conversions']

                            fig = make_subplots(rows=2, cols=1, 
                                              subplot_titles=[f'{m.title()} Comparison' for m in metrics_to_plot],
                                              vertical_spacing=0.1)

                            for idx, metric in enumerate(metrics_to_plot):
                                # Baseline
                                fig.add_trace(go.Scatter(
                                    x=baseline_daily['date'],
                                    y=baseline_daily[metric],
                                    mode='lines',
                                    name=f'Baseline {metric}',
                                    line=dict(color='#667eea', width=2),
                                    showlegend=(idx == 0)
                                ), row=idx+1, col=1)

                                # Campaign
                                fig.add_trace(go.Scatter(
                                    x=campaign_daily['date'],
                                    y=campaign_daily[metric],
                                    mode='lines',
                                    name=f'Campaign {metric}',
                                    line=dict(color='#764ba2', width=2),
                                    showlegend=(idx == 0)
                                ), row=idx+1, col=1)

                                # Add campaign start line
                                fig.add_vline(x=camp_start, line_dash="dash", 
                                            line_color="red", row=idx+1, col=1)

                            fig.update_layout(height=600, hovermode='x unified')
                            st.plotly_chart(fig, use_container_width=True)

                            # Calculate and display lift metrics
                            baseline_avg_rev = baseline_daily['revenue'].mean()
                            campaign_avg_rev = campaign_daily['revenue'].mean()
                            revenue_lift = ((campaign_avg_rev - baseline_avg_rev) / baseline_avg_rev * 100) if baseline_avg_rev > 0 else 0

                            baseline_avg_conv = baseline_daily['conversions'].mean()
                            campaign_avg_conv = campaign_daily['conversions'].mean()
                            conv_lift = ((campaign_avg_conv - baseline_avg_conv) / baseline_avg_conv * 100) if baseline_avg_conv > 0 else 0

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Baseline Avg Revenue", f"${baseline_avg_rev:,.0f}")
                            with col2:
                                st.metric("Campaign Avg Revenue", f"${campaign_avg_rev:,.0f}", 
                                        delta=f"{revenue_lift:+.1f}%")
                            with col3:
                                st.metric("Baseline Avg Conversions", f"{baseline_avg_conv:,.0f}")
                            with col4:
                                st.metric("Campaign Avg Conversions", f"{campaign_avg_conv:,.0f}", 
                                        delta=f"{conv_lift:+.1f}%")

            else:
                st.info("Campaign data not found. Please ensure your data includes campaign information.")
        except Exception as e:
            st.error("Error in campaign analysis. Please check your data.")

    # Tab 3: Enhanced Root Cause Analysis
    with tabs[2]:
        st.markdown("## 🔍 Advanced Root Cause Analysis")
        st.markdown("""
        <div class="analysis-step">
            <h4>Follow our comprehensive analysis workflow to identify root causes of performance issues</h4>
        </div>
        """, unsafe_allow_html=True)

        # Step 1: Configure Analysis
        st.markdown("### Step 1: Configure Analysis Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Select KPI to analyze - exclude conversions as a target if it's a feature
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            kpi_options = ['revenue', 'roas', 'ctr', 'cost', 'profit'] if 'revenue' in df.columns and 'cost' in df.columns else numeric_cols

            # Add profit as an option if we have revenue and cost
            if 'revenue' in df.columns and 'cost' in df.columns:
                df['profit'] = df['revenue'] - df['cost']
                if 'profit' not in kpi_options:
                    kpi_options.insert(0, 'profit')

            available_kpis = [kpi for kpi in kpi_options if kpi in df.columns or kpi == 'profit']

            if available_kpis:
                kpi = st.selectbox("Select Target KPI", available_kpis)
            else:
                kpi = st.selectbox("Select Target KPI", numeric_cols[:5] if numeric_cols else ['revenue'])

        with col2:
            analysis_period = st.selectbox(
                "Analysis Period",
                ["Last 30 days", "Last 90 days", "All time", "Custom"]
            )

            if analysis_period == "Custom" and 'date' in df.columns:
                date_range = st.date_input(
                    "Select date range",
                    value=(df['date'].min(), df['date'].max()),
                    min_value=df['date'].min(),
                    max_value=df['date'].max()
                )

        with col3:
            anomaly_method = st.selectbox(
                "Anomaly Detection Method",
                ["Isolation Forest", "Statistical (Z-score)"]
            )

        # Run Analysis Button
        if st.button("🚀 Run Complete Root Cause Analysis", use_container_width=True):
            with st.spinner("Running comprehensive analysis..."):
                try:
                    # Filter data based on period
                    if analysis_period == "Last 30 days":
                        analysis_df = df[df['date'] >= df['date'].max() - timedelta(days=30)]
                    elif analysis_period == "Last 90 days":
                        analysis_df = df[df['date'] >= df['date'].max() - timedelta(days=90)]
                    elif analysis_period == "Custom":
                        analysis_df = df[(df['date'] >= pd.to_datetime(date_range[0])) & 
                                        (df['date'] <= pd.to_datetime(date_range[1]))]
                    else:
                        analysis_df = df.copy()

                    # Store results
                    st.session_state.rca_results = {
                        'kpi': kpi,
                        'df': analysis_df,
                        'completed': True
                    }
                except Exception as e:
                    st.error("Error running analysis. Please check your data and try again.")
                    st.session_state.rca_results = {'completed': False}

        # Display Results
        if st.session_state.get('rca_results', {}).get('completed', False):
            results = st.session_state.rca_results
            analysis_df = results['df']
            kpi = results['kpi']

            # Create analysis tabs
            analysis_tabs = st.tabs([
                "📊 Anomaly Detection",
                "📈 Correlation Analysis", 
                "🔄 Regression Analysis",
                "🎯 Causal Inference",
                "🏆 Feature Importance",
                "💡 Final Insights"
            ])

            # Anomaly Detection Tab
            with analysis_tabs[0]:
                st.markdown("### 📊 Anomaly Detection Results")

                try:
                    # Detect anomalies
                    if anomaly_method == "Isolation Forest":
                        anomaly_data = detect_anomalies_advanced(analysis_df, kpi, 'isolation_forest')
                    else:
                        anomaly_data = detect_anomalies_advanced(analysis_df, kpi, 'statistical')

                    if anomaly_data is not None:
                        # Visualize anomalies
                        fig = go.Figure()

                        # Normal points
                        normal_data = anomaly_data[~anomaly_data['anomaly']]
                        fig.add_trace(go.Scatter(
                            x=normal_data['date'],
                            y=normal_data['mean'],
                            mode='lines+markers',
                            name='Normal',
                            line=dict(color='#667eea', width=2)
                        ))

                        # Anomaly points
                        anomaly_points = anomaly_data[anomaly_data['anomaly']]
                        if len(anomaly_points) > 0:
                            fig.add_trace(go.Scatter(
                                x=anomaly_points['date'],
                                y=anomaly_points['mean'],
                                mode='markers',
                                name='Anomalies',
                                marker=dict(
                                    color='red',
                                    size=12,
                                    symbol='x',
                                    line=dict(width=2)
                                )
                            ))

                        fig.update_layout(
                            title=f"Anomaly Detection for {kpi.upper()}",
                            xaxis_title="Date",
                            yaxis_title=f"Average {kpi}",
                            height=400,
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Anomaly summary
                        if len(anomaly_points) > 0:
                            st.markdown(f"""
                            <div class="insight-card">
                                <h4>🔍 Anomaly Summary</h4>
                                <ul>
                                    <li>Total anomalies detected: <b>{len(anomaly_points)}</b></li>
                                    <li>Percentage of days with anomalies: <b>{len(anomaly_points)/len(anomaly_data)*100:.1f}%</b></li>
                                    <li>Most severe drop: <b>{anomaly_points['mean'].min():.0f}</b> on {anomaly_points.loc[anomaly_points['mean'].idxmin(), 'date'].strftime('%Y-%m-%d')}</li>
                                    <li>Highest spike: <b>{anomaly_points['mean'].max():.0f}</b> on {anomaly_points.loc[anomaly_points['mean'].idxmax(), 'date'].strftime('%Y-%m-%d')}</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                            # Store anomaly dates for causal analysis
                            st.session_state.rca_results['anomaly_dates'] = anomaly_points['date'].tolist()
                        else:
                            st.info("No anomalies detected in the selected period.")
                except Exception as e:
                    st.error("Error in anomaly detection. Please try a different method or check your data.")

            # Enhanced Correlation Analysis Tab
            with analysis_tabs[1]:
                st.markdown("### 📈 Enhanced Correlation Analysis")

                try:
                    # Variable selection for custom correlation
                    st.markdown("#### Custom Correlation Matrix")

                    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()

                    # Multi-select for variables
                    selected_vars = st.multiselect(
                        "Select variables for correlation analysis",
                        numeric_cols,
                        default=[kpi] + numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
                    )

                    if len(selected_vars) >= 2:
                        # Calculate correlation matrix
                        corr_matrix = analysis_df[selected_vars].corr()

                        # Create heatmap
                        fig = px.imshow(
                            corr_matrix,
                            color_continuous_scale='RdBu_r',
                            title="Custom Correlation Heatmap",
                            labels=dict(color="Correlation"),
                            aspect='auto',
                            text_auto='.2f'
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)

                        # Correlation table with p-values
                        st.markdown("#### Detailed Correlation Analysis with Statistical Significance")

                        # Calculate all correlations with target
                        corr_results = correlation_analysis(analysis_df, kpi)
                        # save for Final Insights
                        st.session_state.rca_results['corr_results'] = corr_results


                        if corr_results is not None and len(corr_results) > 0:
                            # Display as styled dataframe
                            styled_corr = corr_results.style.format({
                                'correlation': '{:.3f}',
                                'p_value': '{:.4f}'
                            }).background_gradient(subset=['correlation'], cmap='RdBu_r', vmin=-1, vmax=1)

                            # Add significance indicators
                            def highlight_significant(row):
                                return ['background-color: #90EE90' if row['p_value'] < 0.05 else '' for _ in row]

                            styled_corr = styled_corr.apply(highlight_significant, axis=1)

                            st.dataframe(styled_corr, use_container_width=True)

                            # Scatter plots for top correlations
                            st.markdown("#### Top Correlations Visualized")

                            top_corr = corr_results[corr_results['significant']].head(4)

                            if len(top_corr) > 0:
                                cols = st.columns(2)
                                for idx, (_, row) in enumerate(top_corr.iterrows()):
                                    with cols[idx % 2]:
                                        fig = px.scatter(
                                            analysis_df, 
                                            x=row['feature'], 
                                            y=kpi,
                                            trendline="ols",
                                            title=f"{kpi} vs {row['feature']} (r={row['correlation']:.3f})",
                                            color='channel' if 'channel' in analysis_df.columns else None
                                        )
                                        fig.update_layout(height=300)
                                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select at least 2 variables for correlation analysis.")

                except Exception as e:
                    st.error("Error in correlation analysis.")

            # Enhanced Regression Analysis Tab
            with analysis_tabs[2]:
                st.markdown("### 🔄 Advanced Regression Analysis")

                try:
                    # Prepare data for regression
                    numeric_df = analysis_df.select_dtypes(include=[np.number]).dropna()

                    # Get available features (excluding target)
                    available_features = [col for col in numeric_df.columns if col != kpi]

                    # Feature selection for regression
                    st.markdown("#### Feature Selection for Regression")

                    col1, col2 = st.columns(2)

                    with col1:
                        # Manual feature selection
                        selected_features = st.multiselect(
                            "Select features for regression model",
                            available_features,
                            default=available_features[:5] if len(available_features) > 5 else available_features
                        )

                    with col2:
                        # Model type selection
                        model_type = st.selectbox(
                            "Select regression type",
                            ["Linear Regression", "Multivariate Regression with Interactions"]
                        )

                    if len(selected_features) >= 1:
                        X = numeric_df[selected_features]
                        y = numeric_df[kpi]

                        # Add constant
                        X_with_const = sm.add_constant(X)

                        if model_type == "Multivariate Regression with Interactions":
                            # Add interaction terms for top 2 features
                            if len(selected_features) >= 2:
                                X_with_const[f'{selected_features[0]}_x_{selected_features[1]}'] = (
                                    X[selected_features[0]] * X[selected_features[1]]
                                )

                        # Fit OLS model
                        model = sm.OLS(y, X_with_const).fit()

                        # Display results
                        st.markdown("#### Regression Results Summary")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("R-squared", f"{model.rsquared:.3f}")
                            st.metric("Adjusted R-squared", f"{model.rsquared_adj:.3f}")

                        with col2:
                            st.metric("F-statistic", f"{model.fvalue:.2f}")
                            st.metric("Prob (F-statistic)", f"{model.f_pvalue:.4f}")

                        with col3:
                            st.metric("AIC", f"{model.aic:.0f}")
                            st.metric("BIC", f"{model.bic:.0f}")

                        # Coefficient interpretation
                        st.markdown("#### Coefficient Interpretation")

                        coef_df = pd.DataFrame({
                            'Feature': model.params.index,
                            'Coefficient': model.params.values,
                            'Std Error': model.bse.values,
                            'P-value': model.pvalues.values,
                            'Significant': model.pvalues.values < 0.05
                        })

                        # Style the dataframe
                        styled_coef = coef_df.style.format({
                            'Coefficient': '{:.4f}',
                            'Std Error': '{:.4f}',
                            'P-value': '{:.4f}'
                        }).apply(lambda x: ['background-color: #90EE90' if x['Significant'] else '' for _ in x], axis=1)

                        st.dataframe(styled_coef, use_container_width=True)

                        # Interpretation guide
                        st.markdown("""
                        <div class="insight-card">
                            <h4>📚 How to Interpret the Results:</h4>
                            <ul>
                                <li><b>R-squared:</b> {:.1f}% of the variance in {kpi} is explained by the model</li>
                                <li><b>Coefficients:</b> For each unit increase in a feature, {kpi} changes by the coefficient value</li>
                                <li><b>P-values:</b> Values < 0.05 indicate statistically significant relationships</li>
                                <li><b>Green highlights:</b> Statistically significant predictors</li>
                            </ul>
                        </div>
                        """.format(model.rsquared * 100, kpi=kpi), unsafe_allow_html=True)

                        # Residual analysis
                        st.markdown("#### Model Diagnostics")

                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Residuals vs Fitted', 'Q-Q Plot', 
                                          'Scale-Location', 'Residuals vs Leverage')
                        )

                        # Residuals vs Fitted
                        fig.add_trace(
                            go.Scatter(
                                x=model.fittedvalues,
                                y=model.resid,
                                mode='markers',
                                marker=dict(color='#667eea'),
                                name='Residuals'
                            ),
                            row=1, col=1
                        )

                        # Q-Q plot
                        qq_data = stats.probplot(model.resid, dist="norm")
                        fig.add_trace(
                            go.Scatter(
                                x=qq_data[0][0],
                                y=qq_data[0][1],
                                mode='markers',
                                marker=dict(color='#764ba2'),
                                name='Q-Q'
                            ),
                            row=1, col=2
                        )

                        # Scale-Location
                        fig.add_trace(
                            go.Scatter(
                                x=model.fittedvalues,
                                y=np.sqrt(np.abs(model.resid_pearson)),
                                mode='markers',
                                marker=dict(color='#9f7aea'),
                                name='Scale-Location'
                            ),
                            row=2, col=1
                        )

                        # Calculate leverage
                        influence = model.get_influence()
                        leverage = influence.hat_matrix_diag

                        # Residuals vs Leverage
                        fig.add_trace(
                            go.Scatter(
                                x=leverage,
                                y=model.resid_pearson,
                                mode='markers',
                                marker=dict(color='#b794f4'),
                                name='Leverage'
                            ),
                            row=2, col=2
                        )

                        fig.update_layout(height=800, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                        # VIF for multicollinearity
                        if len(selected_features) > 1:
                            st.markdown("#### Multicollinearity Check (VIF)")
                            try:
                                vif_data = pd.DataFrame()
                                vif_data["Feature"] = X.columns
                                vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                                                  for i in range(len(X.columns))]

                                # Interpret VIF
                                vif_data['Interpretation'] = vif_data['VIF'].apply(
                                    lambda x: 'No multicollinearity' if x < 5 
                                    else 'Moderate multicollinearity' if x < 10 
                                    else 'High multicollinearity'
                                )

                                styled_vif = vif_data.style.background_gradient(
                                    cmap='RdYlGn_r', subset=['VIF'], vmin=1, vmax=10
                                )
                                st.dataframe(styled_vif, use_container_width=True)

                                if vif_data['VIF'].max() > 10:
                                    st.warning("⚠️ High multicollinearity detected! Consider removing correlated features.")
                            except:
                                st.info("Unable to calculate VIF")

                        # Model summary text
                        with st.expander("📄 Full Statistical Summary"):
                            st.text(model.summary())
                    else:
                        st.warning("Please select at least one feature for regression analysis")

                except Exception as e:
                    st.error(f"Error in regression analysis: {str(e)}")

            # Causal Inference Tab
            with analysis_tabs[3]:
                st.markdown("### 🎯 Causal Inference Analysis")

                try:
                    # Check if we have anomaly dates
                    if 'anomaly_dates' in st.session_state.rca_results and len(st.session_state.rca_results['anomaly_dates']) > 0:

                        st.markdown("#### Interrupted Time Series (ITS) Analysis")

                        # Select intervention date
                        intervention_date = st.selectbox(
                            "Select intervention date (anomaly to analyze)",
                            st.session_state.rca_results['anomaly_dates']
                        )

                        # Run ITS analysis
                        its_model, its_data = causal_inference_its(analysis_df, intervention_date, kpi)

                        if its_model is not None and its_data is not None:
                            # Visualize ITS
                            fig = go.Figure()

                            # Pre-intervention
                            pre_data = its_data[its_data['date'] < intervention_date]
                            fig.add_trace(go.Scatter(
                                x=pre_data['date'],
                                y=pre_data[kpi],
                                mode='lines+markers',
                                name='Pre-intervention',
                                line=dict(color='#667eea', width=2)
                            ))

                            # Post-intervention
                            post_data = its_data[its_data['date'] >= intervention_date]
                            fig.add_trace(go.Scatter(
                                x=post_data['date'],
                                y=post_data[kpi],
                                mode='lines+markers',
                                name='Post-intervention',
                                line=dict(color='#764ba2', width=2)
                            ))

                            # Add intervention line - Fixed timestamp issue
                            fig.add_vline(
                                x=int(intervention_date.timestamp() * 1000),  # Convert to milliseconds
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Intervention"
                            )

                            # Add fitted values
                            its_data['fitted'] = its_model.predict()
                            fig.add_trace(go.Scatter(
                                x=its_data['date'],
                                y=its_data['fitted'],
                                mode='lines',
                                name='Fitted',
                                line=dict(color='green', width=2, dash='dash')
                            ))

                            fig.update_layout(
                                title=f"Interrupted Time Series Analysis - {kpi}",
                                xaxis_title="Date",
                                yaxis_title=kpi,
                                height=400
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # ITS Results
                            st.markdown(f"""
                            <div class="causal-card">
                                <h4>📊 ITS Results</h4>
                                <ul>
                                    <li>Immediate impact: <b>{its_model.params['intervention']:.2f}</b> ({its_model.pvalues['intervention']:.4f} p-value)</li>
                                    <li>Trend change: <b>{its_model.params['time_after']:.2f}</b> ({its_model.pvalues['time_after']:.4f} p-value)</li>
                                    <li>Model R²: <b>{its_model.rsquared:.3f}</b></li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)

                        # Difference-in-Differences
                        if 'device_type' in analysis_df.columns:
                            st.markdown("#### Difference-in-Differences (DiD) Analysis")

                            col1, col2 = st.columns(2)

                            with col1:
                                treatment_group = st.selectbox(
                                    "Treatment Group",
                                    analysis_df['device_type'].unique()
                                )

                            with col2:
                                control_group = st.selectbox(
                                    "Control Group",
                                    [g for g in analysis_df['device_type'].unique() if g != treatment_group]
                                )

                            if st.button("Run DiD Analysis"):
                                did_model, did_data = difference_in_differences(
                                    analysis_df, 
                                    treatment_group, 
                                    control_group, 
                                    intervention_date, 
                                    kpi, 
                                    'device_type'
                                )

                                if did_model is not None and did_data is not None:
                                    # Visualize DiD
                                    fig = go.Figure()

                                    for group in [treatment_group, control_group]:
                                        group_data = did_data[did_data['device_type'] == group]
                                        fig.add_trace(go.Scatter(
                                            x=group_data['date'],
                                            y=group_data[kpi],
                                            mode='lines+markers',
                                            name=group,
                                            line=dict(width=2)
                                        ))

                                    # Fixed timestamp issue for vline
                                    fig.add_vline(
                                        x=int(intervention_date.timestamp() * 1000),
                                        line_dash="dash",
                                        line_color="red",
                                        annotation_text="Treatment Date"
                                    )

                                    fig.update_layout(
                                        title="Difference-in-Differences Analysis",
                                        xaxis_title="Date",
                                        yaxis_title=kpi,
                                        height=400
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                    # DiD coefficient
                                    did_coef = did_model.params['did']
                                    did_pvalue = did_model.pvalues['did']

                                    if did_pvalue < 0.05:
                                        st.success(f"""
                                        ✅ **Significant causal effect detected!**

                                        The treatment had a {did_coef:.2f} effect on {kpi} 
                                        (p-value: {did_pvalue:.4f})
                                        """)
                                    else:
                                        st.info(f"""
                                        ℹ️ **No significant causal effect**

                                        DiD coefficient: {did_coef:.2f} (p-value: {did_pvalue:.4f})
                                        """)
                    else:
                        st.info("Run anomaly detection first to identify intervention dates for causal analysis")
                except Exception as e:
                    st.error("Error in causal analysis. Please check your data.")

            # Enhanced Feature Importance Tab
            with analysis_tabs[4]:
                st.markdown("### 🏆 Feature Importance Analysis")

                try:
                    # Random Forest feature importance
                    rf_model, importance_df = feature_importance_analysis(analysis_df, kpi)

                    if rf_model is not None and len(importance_df) > 0:
                        # Visualize feature importance
                        fig = px.bar(
                            importance_df.head(15),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title=f"Top 15 Features Impacting {kpi} (excluding target-related features)",
                            labels={'importance': 'Feature Importance', 'feature': 'Feature'},
                            color='importance',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)

                        # Feature importance interpretation
                        st.markdown("""
                        <div class="insight-card">
                            <h4>📊 Understanding Feature Importance:</h4>
                            <ul>
                                <li>Higher values indicate features that better predict {}</li>
                                <li>Features related to the target variable (like conversion_rate for conversions) are excluded</li>
                                <li>Use these insights to focus optimization efforts on high-impact areas</li>
                            </ul>
                        </div>
                        """.format(kpi), unsafe_allow_html=True)

                        # Directional Impact Analysis
                        st.markdown("#### Directional Impact Analysis")

                        # Select top features for detailed analysis
                        top_features = importance_df.head(6)['feature'].tolist()

                        if len(top_features) > 0:
                            # Create subplot grid
                            n_features = min(len(top_features), 6)
                            fig = make_subplots(
                                rows=2, cols=3,
                                subplot_titles=[f[:20] + '...' if len(f) > 20 else f for f in top_features[:6]]
                            )

                            for idx, feature in enumerate(top_features[:6]):
                                if feature in analysis_df.columns:
                                    row = idx // 3 + 1
                                    col = idx % 3 + 1

                                    try:
                                        # Create bins for the feature
                                        feature_data = analysis_df[[feature, kpi]].dropna()

                                        if len(feature_data) > 10:
                                            # Create quantile bins
                                            feature_data['bin'] = pd.qcut(
                                                feature_data[feature], 
                                                q=5, 
                                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                                                duplicates='drop'
                                            )

                                            # Calculate mean KPI by bin
                                            bin_means = feature_data.groupby('bin')[kpi].mean()

                                            fig.add_trace(
                                                go.Bar(
                                                    x=bin_means.index,
                                                    y=bin_means.values,
                                                    marker_color='#667eea',
                                                    showlegend=False
                                                ),
                                                row=row, col=col
                                            )

                                            # Update y-axis label
                                            fig.update_yaxes(title_text=f"Avg {kpi}", row=row, col=col)
                                    except:
                                        pass

                            fig.update_layout(height=600, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)

                            # Permutation importance for more robust analysis
                            st.markdown("#### Permutation Importance Analysis")

                            with st.spinner("Calculating permutation importance..."):
                                # Prepare data
                                feature_cols = list(rf_model.feature_names_in_)
                                X_subset = analysis_df[feature_cols].dropna()
                                y_subset = analysis_df.loc[X_subset.index, kpi]

                              
                                perm_importance = permutation_importance( 
                                    rf_model, X_subset, y_subset, n_repeats=10, random_state=42
                                )

                                # Create visualization
                                perm_df = pd.DataFrame({
                                    'feature': feature_cols,
                                    'importance_mean': perm_importance.importances_mean,
                                    'importance_std': perm_importance.importances_std
                                }).sort_values('importance_mean', ascending=False)

                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=perm_df['feature'],
                                    y=perm_df['importance_mean'],
                                    error_y=dict(
                                        type='data',
                                        array=perm_df['importance_std']
                                    ),
                                    marker_color='#764ba2'
                                ))

                                fig.update_layout(
                                    title="Permutation Feature Importance with Uncertainty",
                                    xaxis_title="Feature",
                                    yaxis_title="Importance",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Unable to calculate feature importance. Check if you have enough data.")
                except Exception as e:
                    st.error(f"Error in feature importance analysis: {str(e)}")

            # Final Insights Tab
            with analysis_tabs[5]:
                st.markdown("### 💡 Root Cause Analysis Summary")

                # Compile all insights
                insights = []

                # From anomaly detection
                if 'anomaly_dates' in st.session_state.rca_results:
                    insights.append({
                        'type': 'Anomaly Pattern',
                        'finding': f"Detected {len(st.session_state.rca_results['anomaly_dates'])} anomalies in {kpi}",
                        'impact': 'High',
                        'recommendation': 'Investigate specific dates for external factors or system issues'
                    })

                # From correlation analysis
                corr_results = st.session_state.rca_results.get('corr_results')
                if corr_results is not None and len(corr_results) > 0:
                    top_corr = corr_results.iloc[0]
                    insights.append({
                        'type': 'Key Driver',
                        'finding': f"{top_corr['feature']} shows strongest correlation ({top_corr['correlation']:.3f}) with {kpi}",
                        'impact': 'High',
                        'recommendation': f"Focus optimization efforts on improving {top_corr['feature']}"
                    })

                # From feature importance
                if importance_df is not None and len(importance_df) > 0:
                    top_feature = importance_df.iloc[0]
                    insights.append({
                        'type': 'Primary Factor',
                        'finding': f"{top_feature['feature']} has highest predictive power for {kpi}",
                        'impact': 'High',
                        'recommendation': f"Monitor and optimize {top_feature['feature']} for maximum impact"
                    })

                # Display insights
                for insight in insights:
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>{insight['type']}</h4>
                        <p><b>Finding:</b> {insight['finding']}</p>
                        <p><b>Impact:</b> {insight['impact']}</p>
                        <p><b>Recommendation:</b> {insight['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Root cause hypothesis
                st.markdown("### 🎯 Root Cause Hypotheses")

                if 'anomaly_dates' in st.session_state.rca_results and len(st.session_state.rca_results['anomaly_dates']) > 0:
                    # Analyze what happened on anomaly dates
                    for anomaly_date in st.session_state.rca_results['anomaly_dates'][:3]:  # Top 3 anomalies
                        anomaly_data = analysis_df[analysis_df['date'] == anomaly_date]

                        if len(anomaly_data) > 0:
                            st.markdown(f"""
                            <div class="causal-card">
                                <h4>📅 Anomaly on {anomaly_date.strftime('%Y-%m-%d')}</h4>
                            """, unsafe_allow_html=True)

                            # Check various dimensions
                            checks = []

                            # Device check
                            if 'device_type' in anomaly_data.columns:
                                device_perf = anomaly_data.groupby('device_type')[kpi].mean()
                                worst_device = device_perf.idxmin()
                                checks.append(f"Worst performing device: **{worst_device}**")

                            # Channel check
                            if 'channel' in anomaly_data.columns:
                                channel_perf = anomaly_data.groupby('channel')[kpi].mean()
                                worst_channel = channel_perf.idxmin()
                                checks.append(f"Worst performing channel: **{worst_channel}**")

                            # Campaign check
                            if 'campaign_name' in anomaly_data.columns:
                                campaign_perf = anomaly_data.groupby('campaign_name')[kpi].mean()
                                worst_campaign = campaign_perf.idxmin()
                                checks.append(f"Worst performing campaign: **{worst_campaign}**")

                            for check in checks:
                                st.markdown(f"• {check}")

                            st.markdown("</div>", unsafe_allow_html=True)

                # Action plan
                st.markdown("""
                <div class="recommendation-card">
                    <h3>📋 Recommended Action Plan</h3>
                    <ol>
                        <li><b>Immediate (This Week):</b> Address anomalies on specific dates - check for technical issues or external events</li>
                        <li><b>Short-term (Next 2 Weeks):</b> Optimize top correlated features through A/B testing</li>
                        <li><b>Medium-term (Next Month):</b> Implement monitoring for high-importance features</li>
                        <li><b>Long-term (Next Quarter):</b> Build predictive models to prevent future issues</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)

                # Export report button
                if st.button("📥 Export Full RCA Report", use_container_width=True):
                    st.info("Report export functionality would be implemented here")

    # Tab 4: Statistical Testing
    with tabs[3]:
        st.markdown("## Statistical Testing Suite")

        test_type = st.selectbox("Select Test Type", ["A/B Test", "Significance Calculator", "Power Analysis"])

        if test_type == "A/B Test":
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Test Configuration")

                try:
                    # Find categorical columns for grouping
                    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                    cat_cols = [col for col in cat_cols if col not in ['date', 'day_of_week', 'month']]

                    if cat_cols:
                        test_variable = st.selectbox("Test Variable", cat_cols)

                        # Find numeric columns for metrics
                        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        test_metric = st.selectbox("Success Metric", num_cols)

                        # Select groups
                        groups = df[test_variable].unique()
                        if len(groups) >= 2:
                            group_a = st.selectbox("Control Group", groups)
                            remaining_groups = [g for g in groups if g != group_a]
                            group_b = st.selectbox("Test Group", remaining_groups)

                        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
                    else:
                        st.warning("No categorical columns found for grouping")
                except Exception as e:
                    st.error("Error configuring test. Please check your data.")

            with col2:
                if cat_cols and st.button("🧪 Run Test", use_container_width=True):
                    try:
                        # Get data for both groups
                        group_a_data = df[df[test_variable] == group_a][test_metric].dropna()
                        group_b_data = df[df[test_variable] == group_b][test_metric].dropna()

                        if len(group_a_data) > 0 and len(group_b_data) > 0:
                            # Perform t-test
                            t_stat, p_value = stats.ttest_ind(group_a_data, group_b_data)

                            # Calculate effect size (Cohen's d)
                            mean_a = group_a_data.mean()
                            mean_b = group_b_data.mean()
                            std_a = group_a_data.std()
                            std_b = group_b_data.std()
                            pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
                            cohen_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0

                            # Calculate lift
                            lift = ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0

                            # Display results
                            st.markdown("### Test Results")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    f"{group_a} (Control)",
                                    f"{mean_a:.3f}",
                                    f"n={len(group_a_data)}"
                                )

                            with col2:
                                st.metric(
                                    f"{group_b} (Test)",
                                    f"{mean_b:.3f}",
                                    f"n={len(group_b_data)}"
                                )

                            with col3:
                                st.metric(
                                    "Lift",
                                    f"{lift:+.1f}%",
                                    "Significant" if p_value < (1-confidence_level) else "Not Significant"
                                )

                            # Additional metrics
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("P-value", f"{p_value:.4f}")

                            with col2:
                                st.metric("T-statistic", f"{t_stat:.3f}")

                            with col3:
                                effect_size_interpretation = (
                                    "Small" if abs(cohen_d) < 0.5 
                                    else "Medium" if abs(cohen_d) < 0.8 
                                    else "Large"
                                )
                                st.metric("Cohen's d", f"{cohen_d:.3f}", 
                                        delta=effect_size_interpretation)

                            # Visualization
                            fig = make_subplots(rows=1, cols=2,
                                              subplot_titles=('Distribution Comparison', 'Mean with CI'))

                            # Box plot
                            fig.add_trace(go.Box(
                                y=group_a_data, 
                                name=group_a, 
                                marker_color='#667eea'
                            ), row=1, col=1)

                            fig.add_trace(go.Box(
                                y=group_b_data, 
                                name=group_b, 
                                marker_color='#764ba2'
                            ), row=1, col=1)

                            # Confidence intervals
                            ci_a = 1.96 * std_a / np.sqrt(len(group_a_data))
                            ci_b = 1.96 * std_b / np.sqrt(len(group_b_data))

                            fig.add_trace(go.Bar(
                                x=[group_a, group_b],
                                y=[mean_a, mean_b],
                                error_y=dict(
                                    type='data',
                                    array=[ci_a, ci_b],
                                    visible=True
                                ),
                                marker_color=['#667eea', '#764ba2']
                            ), row=1, col=2)

                            fig.update_layout(
                                height=400,
                                showlegend=False
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Interpretation
                            if p_value < (1-confidence_level):
                                st.success(f"""
                                ✅ **Statistically Significant!**

                                With {confidence_level*100:.0f}% confidence, {group_b} performs 
                                {lift:+.1f}% {'better' if lift > 0 else 'worse'} than {group_a}.

                                - P-value: {p_value:.4f} (< {1-confidence_level:.2f})
                                - Effect size: {effect_size_interpretation} ({cohen_d:.3f})
                                - Minimum detectable effect at this sample size: ~{abs(cohen_d * pooled_std):.2f}
                                """)
                            else:
                                st.info(f"""
                                ℹ️ **No Significant Difference**

                                The test did not find a statistically significant difference 
                                between the groups at {confidence_level*100:.0f}% confidence level.

                                - P-value: {p_value:.4f} (> {1-confidence_level:.2f})
                                - Observed difference: {lift:+.1f}%
                                - Consider increasing sample size or test duration
                                """)

                            # Sample size recommendation
                            if p_value > (1-confidence_level):
                                # Calculate required sample size for 80% power
                                effect_size_target = 0.5  # Medium effect
                                from statsmodels.stats.power import tt_ind_solve_power

                                required_n = tt_ind_solve_power(
                                    effect_size=effect_size_target,
                                    alpha=1-confidence_level,
                                    power=0.8,
                                    ratio=1
                                )

                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>📊 Sample Size Recommendation</h4>
                                    <p>To detect a medium effect size (d=0.5) with 80% power:</p>
                                    <ul>
                                        <li>Required sample size per group: <b>{int(required_n)}</b></li>
                                        <li>Current sample sizes: {len(group_a_data)} (control), {len(group_b_data)} (test)</li>
                                        <li>Additional needed: {max(0, int(required_n) - len(group_a_data))} (control), 
                                            {max(0, int(required_n) - len(group_b_data))} (test)</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("Insufficient data for selected groups")
                    except Exception as e:
                        st.error(f"Error running A/B test: {str(e)}")

        elif test_type == "Power Analysis":
            st.markdown("### Statistical Power Calculator")

            col1, col2 = st.columns(2)

            with col1:
                effect_size = st.slider("Expected Effect Size (Cohen's d)", 0.1, 2.0, 0.5, 0.1)
                alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
                power = st.slider("Desired Power (1-β)", 0.70, 0.95, 0.80, 0.05)

            with col2:
                from statsmodels.stats.power import tt_ind_solve_power

                # Calculate required sample size
                n_required = tt_ind_solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    ratio=1
                )

                st.metric("Required Sample Size (per group)", f"{int(np.ceil(n_required))}")

                # Create power curve
                sample_sizes = np.arange(10, 500, 10)
                powers = []

                for n in sample_sizes:
                    power_calc = tt_ind_solve_power(
                        effect_size=effect_size,
                        nobs1=n,
                        alpha=alpha,
                        ratio=1,
                        alternative='two-sided'
                    )
                    powers.append(power_calc)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sample_sizes,
                    y=powers,
                    mode='lines',
                    line=dict(color='#667eea', width=3)
                ))

                fig.add_hline(y=power, line_dash="dash", line_color="red",
                            annotation_text=f"Target Power: {power}")

                fig.add_vline(x=n_required, line_dash="dash", line_color="green",
                            annotation_text=f"Required n: {int(n_required)}")

                fig.update_layout(
                    title="Statistical Power vs Sample Size",
                    xaxis_title="Sample Size (per group)",
                    yaxis_title="Statistical Power",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

    # Tab 5: AI Recommendations with Chat Interface
    with tabs[4]:
        st.markdown("## 🤖 AI-Powered Recommendations")

        # Generate recommendations based on actual data patterns
        try:
            recommendations = []

            # Channel performance recommendation
            if 'channel' in df.columns and 'roas' in df.columns:
                channel_perf = df.groupby('channel')['roas'].mean().sort_values(ascending=False)
                if len(channel_perf) > 1:
                    best_channel = channel_perf.index[0]
                    worst_channel = channel_perf.index[-1]
                    recommendations.append({
                        'icon': '📈',
                        'title': 'Channel Optimization',
                        'text': f"**{best_channel}** shows {channel_perf.iloc[0]:.1f}x ROAS. "
                               f"Consider reallocating budget from **{worst_channel}** ({channel_perf.iloc[-1]:.1f}x ROAS).",
                        'impact': 'High',
                        'effort': 'Low'
                    })

            # Device optimization
            if 'device_type' in df.columns and 'conversion_rate' in df.columns:
                device_conv = df.groupby('device_type')['conversion_rate'].mean()
                if 'Mobile' in device_conv.index and 'Desktop' in device_conv.index:
                    mobile_rate = device_conv['Mobile']
                    desktop_rate = device_conv['Desktop']
                    if mobile_rate < desktop_rate * 0.8:
                        recommendations.append({
                            'icon': '📱',
                            'title': 'Mobile Optimization',
                            'text': f"Mobile conversion rate ({mobile_rate:.1%}) is "
                                   f"{((desktop_rate - mobile_rate) / desktop_rate * 100):.0f}% lower than desktop. "
                                   f"Optimize mobile experience for quick wins.",
                            'impact': 'High',
                            'effort': 'Medium'
                        })

            # Time-based insights
            if 'date' in df.columns and 'conversions' in df.columns:
                df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()
                dow_conv = df.groupby('day_of_week')['conversions'].sum()
                if len(dow_conv) > 0:
                    best_day = dow_conv.idxmax()
                    recommendations.append({
                        'icon': '📅',
                        'title': 'Timing Strategy',
                        'text': f"**{best_day}s** show highest conversion volume. "
                               f"Schedule important campaigns and increase bids on this day.",
                        'impact': 'Medium',
                        'effort': 'Low'
                    })

            # Regional insights
            if 'region' in df.columns and 'revenue' in df.columns:
                region_rev = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
                if len(region_rev) > 1:
                    top_region = region_rev.index[0]
                    recommendations.append({
                        'icon': '🌍',
                        'title': 'Geographic Focus',
                        'text': f"**{top_region}** generates {(region_rev.iloc[0] / region_rev.sum() * 100):.0f}% "
                               f"of total revenue. Consider region-specific campaigns and increased investment.",
                        'impact': 'High',
                        'effort': 'High'
                    })

            # Source/Medium insights
            if 'source_medium' in df.columns and 'profit' in df.columns:
                sm_profit = df.groupby('source_medium')['profit'].sum().sort_values(ascending=False)
                if len(sm_profit) > 1:
                    top_source = sm_profit.index[0]
                    profit_share = sm_profit.iloc[0] / sm_profit.sum() * 100
                    recommendations.append({
                        'icon': '💰',
                        'title': 'Profit Driver Focus',
                        'text': f"**{top_source}** drives {profit_share:.0f}% of total profit. "
                               f"Increase investment and protect this channel from competitors.",
                        'impact': 'High',
                        'effort': 'Low'
                    })

            # Age group insights
            if 'age_group' in df.columns and 'revenue' in df.columns:
                age_revenue = df.groupby('age_group')['revenue'].sum().sort_values(ascending=False)
                if len(age_revenue) > 1:
                    top_age = age_revenue.index[0]
                    recommendations.append({
                        'icon': '👥',
                        'title': 'Demographic Targeting',
                        'text': f"**{top_age}** age group generates highest revenue. "
                               f"Create targeted content and adjust messaging for this demographic.",
                        'impact': 'Medium',
                        'effort': 'Medium'
                    })

            # Display recommendations
            st.markdown("""
            <div class="recommendation-card">
                <h3>🎯 Strategic Recommendations Based on Your Data</h3>
            </div>
            """, unsafe_allow_html=True)

            for i, rec in enumerate(recommendations[:6]):  # Show top 6
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"""
                    ### {rec['icon']} {rec['title']}
                    {rec['text']}
                    """)

                with col2:
                    st.markdown(f"""
                    **Impact:** {rec['impact']}  
                    **Effort:** {rec['effort']}
                    """)

                if i < len(recommendations) - 1:
                    st.markdown("---")

            # Quick wins section
            st.markdown("""
            ### 🎯 Quick Wins - Implement This Week

            1. **Budget Reallocation** - Move 20% of underperforming channel budget to top performer
            2. **Mobile Testing** - Run A/B test on mobile checkout flow
            3. **Day Parting** - Increase bids 25% on high-converting days
            4. **Creative Refresh** - Update ad creatives for campaigns older than 60 days
            5. **Quality Score** - Review and improve ad relevance for low QS campaigns
            6. **Audience Segmentation** - Create separate campaigns for top-performing demographics
            """)

        except Exception as e:
            st.error("Error generating recommendations. Please check your data.")

        # AI Chat Interface
        st.markdown("---")
        st.markdown("### 💬 Ask AI Marketing Assistant")
        st.markdown("""
        <div class="ai-chat-container">
            <p>Ask questions about your marketing data, get personalized recommendations, or explore insights!</p>
        </div>
        """, unsafe_allow_html=True)

        # Create two columns for better layout
        col1, col2 = st.columns([3, 1])

        with col1:
            user_question = st.text_area(
                "Type your question here:",
                placeholder="Example: What's the best channel for customer acquisition? How can I improve mobile conversion rates?",
                height=100,
                key="ai_question"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            ask_button = st.button("🤖 Ask AI", use_container_width=True, type="primary")

        # AI Response handling
        if ask_button and user_question:
            with st.spinner("🤔 AI is thinking..."):
                # Placeholder for AI response
                # In production, you would call your GPT-3.5 API here
                st.markdown("""
                <div class="ai-chat-container">
                    <h4>🤖 AI Assistant Response:</h4>
                    <p><i>AI functionality will be connected once you add your OpenAI API key in the code.</i></p>
                    <p>To enable AI responses:</p>
                    <ol>
                        <li>Add your OpenAI API key to the code</li>
                        <li>Implement the API call with max_tokens=200</li>
                        <li>Process the response and display here</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)

                # Store the conversation
                st.session_state.ai_messages.append({
                    'user': user_question,
                    'assistant': "AI response will appear here once API is connected"
                })

        # Display conversation history
        if st.session_state.ai_messages:
            st.markdown("### 📝 Conversation History")
            for msg in st.session_state.ai_messages[-5:]:  # Show last 5 messages
                st.markdown(f"**You:** {msg['user']}")
                st.markdown(f"**AI:** {msg['assistant']}")
                st.markdown("---")

else:
    # Welcome screen when no data
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>👋 Welcome to MITS!</h2>
        <p style="font-size: 1.2em;">Your AI-powered marketing analytics platform</p>
        <br>
        <p>Get started in seconds:</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Launch Product Tour ", use_container_width=True):
            st.session_state.demo_mode = True
            with st.spinner("Creating amazing demo data..."):
                demo_data = generate_demo_data()
                if demo_data is not None:
                    st.session_state.data = demo_data
                else:
                    st.error("Failed to generate demo data. Please try again.")
            st.rerun()

    st.markdown("---")

    # Feature preview
    st.markdown("### ✨ What You'll Get:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **📊 Executive Dashboard**
        - Real-time KPI tracking
        - Revenue & conversion trends
        - Channel performance matrix
        - Interactive visualizations
        - Custom variable analysis
        - Time series comparisons

        **📈 Campaign Analysis**
        - Multi-campaign comparison
        - Conversion funnel analysis
        - Source/Medium performance
        - Demographic insights
        - Profit analysis by segment
        - Time series campaign analysis

        **🔍 Advanced Root Cause Analysis**
        - Multiple anomaly detection methods
        - Enhanced correlation analysis with custom variables
        - Multivariate regression with interactions
        - Causal inference (ITS, DiD)
        - Feature importance (excluding target variables)
        - Comprehensive insights and recommendations
        """)

    with col2:
        st.markdown("""
        **🧪 Statistical Testing**
        - A/B test calculator with effect size
        - Significance testing
        - Power analysis
        - Sample size recommendations
        - Visual comparisons

        **🤖 AI Recommendations**
        - Smart optimization tips
        - Budget allocation advice
        - Performance predictions
        - Demographic targeting
        - Profit-focused insights
        - AI Chat Assistant (Ready for API)

        **📱 Enhanced Features**
        - Source/Medium analysis
        - Demographic breakdowns
        - Geographic insights
        - Device performance tracking
        - Custom time period analysis
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>MITS - Marketing Intelligence & Testing Suite | Built with ❤️ for data-driven marketers</p>
</div>
""", unsafe_allow_html=True)

