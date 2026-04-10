import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="⚡",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 50%, #faf0ff 100%);
}

/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hero Banner */
.hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    border-radius: 24px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -30%;
    right: 15%;
    width: 250px;
    height: 250px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
}
.hero h1 {
    color: white;
    font-size: 3rem;
    font-weight: 800;
    margin: 0 0 12px 0;
    letter-spacing: -1px;
}
.hero p {
    color: rgba(255,255,255,0.85);
    font-size: 1.15rem;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: white;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 16px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Metric Cards */
.metric-card {
    background: white;
    border-radius: 20px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.1);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-4px); }
.metric-icon {
    font-size: 2rem;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 4px 0;
    letter-spacing: -1px;
}
.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-sub {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* Section Headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e1b4b;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Chart Cards */
.chart-card {
    background: white;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 4px 24px rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.1);
    margin-bottom: 24px;
}

/* Upload Area */
.upload-area {
    background: white;
    border: 2px dashed #c7d2fe;
    border-radius: 20px;
    padding: 48px;
    text-align: center;
    margin: 16px 0 32px 0;
    transition: border-color 0.2s;
}
.upload-area:hover { border-color: #6366f1; }
.upload-icon { font-size: 3rem; margin-bottom: 16px; }
.upload-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1e1b4b;
    margin-bottom: 8px;
}
.upload-sub { color: #94a3b8; font-size: 0.95rem; }

/* How it works */
.how-card {
    background: white;
    border-radius: 20px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.1);
    height: 100%;
}
.how-number {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    font-weight: 800;
    margin: 0 auto 16px auto;
    color: white;
}
.how-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e1b4b;
    margin-bottom: 8px;
}
.how-desc { color: #64748b; font-size: 0.9rem; line-height: 1.6; }

/* Stat badge */
.stat-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 8px;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    margin-top: 16px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: white !important;
    border-right: 1px solid #e2e8f0 !important;
}
.sidebar-logo {
    background: linear-gradient(135deg, #6366f1, #ec4899);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin-bottom: 24px;
    color: white;
    font-size: 1.5rem;
    font-weight: 800;
}
.sidebar-section {
    background: #f8f9ff;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}
.sidebar-section h4 {
    color: #6366f1;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 12px 0;
}
.sidebar-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #e2e8f0;
    font-size: 0.9rem;
    color: #475569;
}
.sidebar-item:last-child { border-bottom: none; }
.sidebar-value {
    font-weight: 700;
    color: #1e1b4b;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
    return model, features

model, features = load_model()

# ── Preprocess ────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    if 'Churn' in df.columns:
        df.drop('Churn', axis=1, inplace=True)
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ ChurnIQ</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="sidebar-section">
            <h4>🤖 Model Info</h4>
            <div class="sidebar-item">
                <span>Algorithm</span>
                <span class="sidebar-value">XGBoost</span>
            </div>
            <div class="sidebar-item">
                <span>Accuracy</span>
                <span class="sidebar-value">~80%</span>
            </div>
            <div class="sidebar-item">
                <span>Features</span>
                <span class="sidebar-value">19</span>
            </div>
            <div class="sidebar-item">
                <span>Dataset</span>
                <span class="sidebar-value">Telco 100K</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="sidebar-section">
            <h4>📖 How to Use</h4>
            <div class="sidebar-item">1. Upload a CSV file</div>
            <div class="sidebar-item">2. View instant predictions</div>
            <div class="sidebar-item">3. Analyze churn factors</div>
            <div class="sidebar-item">4. Download results</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="sidebar-section">
            <h4>📁 Supported Format</h4>
            <div class="sidebar-item">
                <span>File Type</span>
                <span class="sidebar-value">CSV</span>
            </div>
            <div class="sidebar-item">
                <span>Max Size</span>
                <span class="sidebar-value">200MB</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
    <div class="hero">
        <div class="hero-badge">⚡ AI Powered</div>
        <h1>Customer Churn Intelligence</h1>
        <p>Upload your customer data and instantly predict who's about to leave — powered by XGBoost ML</p>
    </div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────
uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

if not uploaded_file:
    st.markdown("""
        <div class="upload-area">
            <div class="upload-icon">📂</div>
            <div class="upload-title">Drop your CSV file here</div>
            <div class="upload-sub">Upload your customer dataset to get instant churn predictions</div>
        </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="section-header">🧠 How It Works</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("#6366f1", "1", "Upload Data", "Upload your customer CSV dataset with usage and billing info"),
        ("#8b5cf6", "2", "AI Analysis", "XGBoost model analyzes 19 features to predict churn risk"),
        ("#ec4899", "3", "Get Insights", "View churn rate, risk factors, and probability distribution"),
        ("#f59e0b", "4", "Download", "Export full predictions with churn scores as CSV"),
    ]
    for col, (color, num, title, desc) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(f"""
                <div class="how-card">
                    <div class="how-number" style="background:{color}">{num}</div>
                    <div class="how-title">{title}</div>
                    <div class="how-desc">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

else:
    df = pd.read_csv(uploaded_file)

    with st.expander("👀 Preview Raw Data", expanded=False):
        st.dataframe(df.head(), use_container_width=True)

    # Predict
    df_processed = preprocess(df)
    df_processed = df_processed.reindex(columns=features, fill_value=0)
    predictions = model.predict(df_processed)
    probabilities = model.predict_proba(df_processed)[:, 1]

    df['Churn Prediction'] = ['🔴 Churn' if p == 1 else '🟢 Stay' for p in predictions]
    df['Churn Probability %'] = [round(p * 100, 1) for p in probabilities]

    total = len(df)
    churned = int(sum(predictions))
    staying = total - churned
    churn_rate = round(churned / total * 100, 1)
    avg_prob = round(float(np.mean(probabilities)) * 100, 1)

    # ── KPI Cards ─────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Overview</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    kpis = [
        ("👥", str(total), "Total Customers", "#6366f1", "Full dataset"),
        ("🔴", str(churned), "At Risk", "#ef4444", f"{churn_rate}% churn rate"),
        ("🟢", str(staying), "Retained", "#10b981", f"{100 - churn_rate}% retention"),
        ("⚡", f"{avg_prob}%", "Avg Risk Score", "#f59e0b", "Mean churn probability"),
    ]
    for col, (icon, value, label, color, sub) in zip([col1, col2, col3, col4], kpis):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value" style="color:{color}">{value}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">🥧 Churn Breakdown</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=['Retained', 'At Risk'],
            values=[staying, churned],
            hole=0.65,
            marker=dict(
                colors=['#10b981', '#ef4444'],
                line=dict(color='white', width=3)
            ),
            textinfo='percent+label',
            textfont=dict(size=13)
        ))
        fig.add_annotation(
            text=f"<b>{churn_rate}%</b><br>Churn",
            x=0.5, y=0.5,
            font=dict(size=18, color='#1e1b4b'),
            showarrow=False
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
            margin=dict(t=20, b=40),
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🔍 Top Risk Factors</div>', unsafe_allow_html=True)
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True).tail(8)

        fig2 = go.Figure(go.Bar(
            x=importance['Importance'],
            y=importance['Feature'],
            orientation='h',
            marker=dict(
                color=importance['Importance'],
                colorscale=[[0, '#c7d2fe'], [0.5, '#8b5cf6'], [1, '#6366f1']],
                line=dict(color='rgba(0,0,0,0)')
            )
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=20, l=10),
            height=320,
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9', color='#94a3b8'),
            yaxis=dict(color='#475569')
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Probability Distribution ──────────────────────────────
    st.markdown('<div class="section-header">📉 Risk Score Distribution</div>', unsafe_allow_html=True)
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=df['Churn Probability %'],
        nbinsx=30,
        marker=dict(
            color=df['Churn Probability %'],
            colorscale=[[0, '#c7d2fe'], [0.5, '#8b5cf6'], [1, '#ef4444']],
            line=dict(color='white', width=1)
        ),
        opacity=0.85
    ))
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='Churn Probability %',
            showgrid=True,
            gridcolor='#f1f5f9',
            color='#94a3b8'
        ),
        yaxis=dict(
            title='Number of Customers',
            showgrid=True,
            gridcolor='#f1f5f9',
            color='#94a3b8'
        ),
        bargap=0.1,
        height=300,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Results Table ─────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Prediction Results</div>', unsafe_allow_html=True)
    result_cols = ['Churn Prediction', 'Churn Probability %'] + \
                  [c for c in df.columns if c not in ['Churn Prediction', 'Churn Probability %']]
    st.dataframe(
        df[result_cols].style.map(
            lambda x: 'color: #ef4444; font-weight: bold' if x == '🔴 Churn'
            else ('color: #10b981; font-weight: bold' if x == '🟢 Stay' else ''),
            subset=['Churn Prediction']
        ),
        use_container_width=True,
        height=400
    )

    # ── Download ──────────────────────────────────────────────
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Full Predictions as CSV",
        data=csv,
        file_name='churn_predictions.csv',
        mime='text/csv'
    )