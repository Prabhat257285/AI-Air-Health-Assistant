import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Air Health Assistant - India Wide", layout="wide")

# ---------- LOAD MODEL ----------
model = joblib.load("best_aqi_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
# ---------- LOAD DATA ----------
try:
    df = pd.read_csv("air_quality_with_state.csv")
except FileNotFoundError:
    df = pd.read_csv("air_quality.csv")
    if "state" not in df.columns:
        df["state"] = "Unknown"

states = sorted(df["state"].unique())

st.markdown("""
<style>

/* ===== Modern Dark Theme Background ===== */
.stApp {
    background: linear-gradient(
        135deg,
        #0f0f1e 0%,
        #1a0033 25%,
        #2d004d 50%,
        #1a0033 75%,
        #0f0f1e 100%
    );
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    color: #ffffff;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ===== Typography ===== */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-shadow: 0 0 20px rgba(168, 85, 247, 0.3);
}

h1 { font-size: 3.5rem; margin-bottom: 1rem; }
h2 { font-size: 2.5rem; margin: 1.5rem 0 1rem 0; }
h3 { font-size: 1.8rem; margin: 1rem 0 0.5rem 0; }

/* ===== Main Container ===== */
.block-container {
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* ===== Glass-Morphism Cards ===== */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    margin: 1.5rem 0;
    transition: all 0.3s ease;
}

.glass-card:hover {
    background: rgba(255, 255, 255, 0.12);
    border-color: rgba(168, 85, 247, 0.4);
    box-shadow: 0 8px 40px rgba(168, 85, 247, 0.2);
    transform: translateY(-2px);
}

/* ===== Neon Button Styling ===== */
.stButton>button {
    width: 100%;
    height: 60px;
    font-size: 1.1rem;
    font-weight: 700;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #f97316 100%);
    color: #ffffff;
    box-shadow: 0 0 30px rgba(168, 85, 247, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    transition: all 0.4s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 50px rgba(168, 85, 247, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.3);
    background: linear-gradient(135deg, #ec4899 0%, #f97316 50%, #fbbf24 100%);
}

.stButton>button:active {
    transform: translateY(0px);
}

/* ===== Input Fields ===== */
input, textarea {
    background-color: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(168, 85, 247, 0.3) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    padding: 12px 16px !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

input:focus, textarea:focus {
    background-color: rgba(255, 255, 255, 0.12) !important;
    border-color: rgba(168, 85, 247, 0.8) !important;
    box-shadow: 0 0 20px rgba(168, 85, 247, 0.3) !important;
}

/* ===== Selectbox Styling ===== */
div[data-baseweb="select"]>div {
    background-color: rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(168, 85, 247, 0.3) !important;
}

div[data-baseweb="select"] span {
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}

div[data-baseweb="menu"] {
    background-color: #1a1a2e !important;
    border: 1px solid rgba(168, 85, 247, 0.3) !important;
}

div[data-baseweb="menu"] span {
    color: #ffffff !important;
}

div[data-baseweb="menu"] li:hover {
    background-color: rgba(168, 85, 247, 0.2) !important;
}

/* ===== Slider Styling ===== */
.stSlider>div>div>div>div {
    background: linear-gradient(90deg, #a855f7, #ec4899) !important;
}

div[data-baseweb="slider"] {
    padding: 20px 0;
}

/* ===== Metric Cards ===== */
.metric-box {
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(236, 72, 153, 0.15));
    border: 1px solid rgba(168, 85, 247, 0.4);
    border-radius: 15px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-box:hover {
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.25), rgba(236, 72, 153, 0.25));
    border-color: rgba(168, 85, 247, 0.8);
    transform: translateY(-3px);
}

/* ===== Expander Styling ===== */
.streamlit-expanderHeader {
    background: rgba(168, 85, 247, 0.15) !important;
    border: 1px solid rgba(168, 85, 247, 0.3) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}

.streamlit-expanderHeader:hover {
    background: rgba(168, 85, 247, 0.25) !important;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 2px solid rgba(168, 85, 247, 0.3);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    color: rgba(255, 255, 255, 0.7);
    padding: 12px 20px;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(168, 85, 247, 0.3), rgba(236, 72, 153, 0.3));
    color: #ffffff;
    box-shadow: 0 2px 8px rgba(168, 85, 247, 0.3);
}

.stTabs [aria-selected="false"]:hover {
    color: rgba(255, 255, 255, 0.9);
}

/* ===== Metric Styling ===== */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.1), rgba(236, 72, 153, 0.1));
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    transition: all 0.3s ease;
}

[data-testid="metric-container"]:hover {
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(236, 72, 153, 0.2));
    border-color: rgba(168, 85, 247, 0.6);
}

/* ===== Alert Boxes ===== */
.stAlert {
    border-radius: 12px;
    border-left: 4px solid;
}

/* ===== Divider ===== */
hr {
    border-color: rgba(168, 85, 247, 0.3);
}

/* ===== DataFrame Styling ===== */
.dataframe {
    background-color: rgba(255, 255, 255, 0.05) !important;
}

.dataframe th {
    background-color: rgba(168, 85, 247, 0.2) !important;
    color: #ffffff !important;
    border-color: rgba(168, 85, 247, 0.3) !important;
}

.dataframe td {
    border-color: rgba(168, 85, 247, 0.1) !important;
    color: #ffffff !important;
}

.dataframe tr:hover {
    background-color: rgba(168, 85, 247, 0.1) !important;
}

/* ===== Scrollbar ===== */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #a855f7, #ec4899);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #ec4899, #f97316);
}

/* ===== Remove default styling ===== */
div.element-container,
div.stMarkdown,
div.stText,
div.stSelectbox,
div.stButton {
    background: transparent !important;
    box-shadow: none !important;
}

/* ===== Column Gap ===== */
.element-container {
    margin-bottom: 1rem;
}

/* ===== Plotly Charts ===== */
.plotly-chart {
    border-radius: 15px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(236, 72, 153, 0.15));
    border: 1px solid rgba(168, 85, 247, 0.4);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
">
    <h1 style="
        margin: 0 0 1rem 0;
        font-size: 3rem;
        background: linear-gradient(135deg, #a855f7, #ec4899, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        letter-spacing: 2px;
    ">
    🌍 AI Air Health Assistant
    </h1>
    <p style="
        margin: 0;
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
        letter-spacing: 0.8px;
    ">
    Real-Time Air Quality Intelligence Across India
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- CITY SELECTION PANEL ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 2], gap="medium")

with col1:
    st.markdown("#### 🗺️ Region")
    state = st.selectbox("Select State", states, label_visibility="collapsed")

with col2:
    st.markdown("#### 🏙️ Location")
    cities_in_state = sorted(df[df["state"] == state]["city"].unique())
    city = st.selectbox("Select City", cities_in_state, label_visibility="collapsed")

with col3:
    st.markdown("#### 📅 Period")
    days_to_forecast = st.slider("Days to Forecast", min_value=1, max_value=14, value=7, label_visibility="collapsed")

st.markdown(
    "<div style='margin-top:20px; padding:15px; background:linear-gradient(90deg,rgba(168,85,247,0.1),rgba(236,72,153,0.1)); border-left:4px solid #a855f7; border-radius:8px; font-size:1rem;'>"
    f"📍 <b>{city}</b> • <span style='opacity:0.8;'>{state}</span> | 📊 <b>{days_to_forecast}-Day</b> Forecast"
    "</div>",
    unsafe_allow_html=True
)

# ---------- PREPARE DATA ----------
city_df = df[df["city"] == city].sort_values("date")

if len(city_df) < 4:
    st.error("Not enough historical data.")
    st.stop()

last3 = city_df["aqi"].tail(3).values
latest = city_df.iloc[-1]
rolling_mean = city_df["aqi"].tail(7).mean()

input_data = {
    "pm25": latest["pm25"],
    "pm10": latest["pm10"],
    "no2": latest["no2"],
    "nh3": latest["nh3"],
    "so2": latest["so2"],
    "co": latest["co"],
    "o3": latest["o3"],
    "temperature": latest["temperature"],
    "humidity": latest["humidity"],
    "aqi_lag1": last3[-1],
    "aqi_lag2": last3[-2],
    "aqi_lag3": last3[-3],
    "aqi_rolling_mean_7": rolling_mean,
    "city_Jaipur": 1 if city=="Jaipur" else 0,
    "city_Lucknow": 1 if city=="Lucknow" else 0
}

features = pd.DataFrame([input_data])
features = features[feature_columns]

features = np.array([[ 
    latest["pm25"],
    latest["pm10"],
    latest["no2"],
    latest["nh3"],              
    latest["so2"],
    latest["co"],
    latest["o3"],
    latest["temperature"],
    latest["humidity"],
    last3[-1],                  
    last3[-2],                  
    last3[-3],                  
    rolling_mean,               
    latest.get("city_Jaipur",0),
    latest.get("city_Lucknow",0)
]])

st.markdown("<br>", unsafe_allow_html=True)

# ---------- AI PREDICTION BUTTON ----------
prediction = None

st.markdown("""
<div class="glass-card">
    <h3 style="margin: 0 0 1rem 0; display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 2rem;">🤖</span>
        <span>AI Prediction Engine</span>
    </h3>
    <p style="opacity: 0.8; margin: 0;">Click below to get a real-time AQI prediction powered by advanced machine learning</p>
</div>
""", unsafe_allow_html=True)

if st.button("🚀 Predict Tomorrow's AQI", use_container_width=True):
    with st.spinner("⏳ AI is analyzing air patterns..."):
        time.sleep(2)
        prediction = int(model.predict(features)[0])


# ---------- ONLY SHOW RESULTS AFTER CLICK ----------
if prediction:

    # ---------- AQI CATEGORY ----------
    def aqi_info(aqi):
        if aqi <= 50: return "GOOD 😊", "#00e676"
        elif aqi <= 100: return "MODERATE 🙂", "#ffd54f"
        elif aqi <= 200: return "UNHEALTHY ⚠️", "#ff9800"
        else: return "VERY DANGEROUS 🚨", "#ff4b4b"

    status, color = aqi_info(prediction)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- HERO ----------
    col1, col2 = st.columns([2,1])

# LEFT PANEL
    with col1:
            st.markdown(f"""
            <div style="
            padding:30px;
            border-radius:18px;
            background:linear-gradient(145deg,#161b22,#0e1117);
            box-shadow:0 0 25px rgba(0,0,0,0.6);
            border-left:8px solid {color};
            ">
            
            <h1 style="
            font-size:44px;
            font-weight:800;
            letter-spacing:2px;
            margin-bottom:6px;
            background:linear-gradient(90deg,#ffffff,#8be9fd);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            ">
            {city}
            </h1>

            <p style="
            font-size:14px;
            opacity:0.65;
            letter-spacing:0.5px;
            margin-bottom:10px;
            ">
            {datetime.now().strftime("%d %B %Y")}
            </p>
            
            <h1 style="
            color:{color};
            margin-top:10px;
            font-size:42px;
            font-weight:800;
            letter-spacing:1px;
            ">
            {status}
            </h1>
            
            <p style="
            opacity:0.8;
            font-size:15px;
            ">
            Real-time AI air quality assessment
            </p>

            </div>
            """, unsafe_allow_html=True)


# RIGHT PANEL — SMOOTH COUNT ANIMATION
    with col2:
        placeholder = st.empty()

        for i in range(0, prediction + 1, max(1, prediction//60)):
            placeholder.markdown(f"""
            <div style="
            padding:30px;
            border-radius:20px;
            text-align:center;
            background:linear-gradient(145deg,#0e1117,#161b22);
            box-shadow:0 0 30px {color}40;">
            
            <div style="
                font-size:75px;
                font-weight:bold;
                color:{color};
                text-shadow:0 0 15px {color};">
                {i}
            </div>

            <h4 style="opacity:0.7;">AQI INDEX</h4>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(0.015)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- AQI BAR ----------
    st.markdown("### 🌡 AQI Severity Meter")

    fig_meter = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        number={'font': {'size': 50}},
        title={'text': "Air Quality Index", 'font': {'size': 22}},
    
        gauge={
            'axis': {'range': [0, 500]},
        
            'bar': {'color': "white"},
        
            'steps': [
            {'range': [0, 50], 'color': "#00E400"},
            {'range': [50, 100], 'color': "#FFFF00"},
            {'range': [100, 200], 'color': "#FF7E00"},
            {'range': [200, 300], 'color': "#FF0000"},
            {'range': [300, 500], 'color': "#8F3F97"}
        ],
        
            'threshold': {
            'line': {'color': "cyan", 'width': 4},
            'thickness': 0.75,
            'value': prediction
        }
        }
    ))

    fig_meter.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=30, r=30, t=60, b=20)
    )

    st.plotly_chart(fig_meter, use_container_width=True)

    # ---------- BEAUTIFUL HEALTH IMPACT ----------
    st.subheader("👨‍👩‍👧‍👦 Health Risk Analysis")

    def risk_card(title, risk, color):
        st.markdown(f"""
    <div style="
        padding:18px;
        border-radius:15px;
        background:linear-gradient(145deg,#161b22,#0e1117);
        border-left:6px solid {color};
        box-shadow:0 0 15px rgba(0,0,0,0.5);
        text-align:center;">
        <h4>{title}</h4>
        <h2 style="color:{color};">{risk}</h2>
    </div>
    """, unsafe_allow_html=True)


    c1, c2, c3, c4 = st.columns(4)

    with c1:
        risk_card("😷 Asthma Patients",
              "HIGH" if prediction > 100 else "LOW",
              "#ff4b4b" if prediction > 100 else "#00e676")

    with c2:
        risk_card("👶 Children",
              "MODERATE" if prediction > 80 else "LOW",
              "#ff9800" if prediction > 80 else "#00e676")

    with c3:
        risk_card("👴 Elderly",
              "HIGH" if prediction > 120 else "LOW",
              "#ff4b4b" if prediction > 120 else "#00e676")

    with c4:
        risk_card("🏃 Healthy Adults",
              "MODERATE" if prediction > 150 else "LOW",
              "#ff9800" if prediction > 150 else "#00e676")

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------AI ADVICE CARD----------
    st.subheader("🤖 AI Health Advice")

    if prediction < 100:
        advice = "🌿 Great day! Enjoy outdoor activities safely."
        bg = "#0f5132"
        border = "#00ff9d"

    elif prediction < 200:
        advice = "😷 Air quality moderate. Limit outdoor exercise & wear a mask."
        bg = "#664d03"
        border = "#ffd43b"

    else:
        advice = "🚨 Dangerous air! Stay indoors & use N95 mask."
        bg = "#58151c"
        border = "#ff4d4d"


    st.markdown(f"""
    <div style="
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, {bg}, #0e1117);
    border-left: 6px solid {border};
    box-shadow: 0 0 30px rgba(0,0,0,0.5);
    font-size: 20px;
    font-weight: 500;
    ">
    {advice}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

   # ---------- BEAUTIFUL TREND GRAPH ----------
    st.subheader("📊 AQI Trend of Last Year 2025")

    past = city_df.tail(30)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=past["date"],
        y=past["aqi"],
        mode="lines+markers",
        line=dict(width=4, color="#00e5ff"),
        marker=dict(size=8, color="#00e5ff"),
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.12)",
        hovertemplate="<b>Date:</b> %{x}<br><b>AQI:</b> %{y}<extra></extra>"
))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),

    title=dict(
        text="Air Quality Trend",
        font=dict(size=22)
    ),

    xaxis=dict(
        showgrid=False,
        title="Date"
    ),

    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        title="AQI Level"
    ),

    plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========== UNIQUE FEATURE 1: POLLUTION CONTRIBUTOR ANALYSIS ==========
    st.subheader("🧪 Pollution Contributor Breakdown")
    
    pollutants_dict = {
        "PM2.5": latest["pm25"],
        "PM10": latest["pm10"],
        "NO₂": latest["no2"],
        "NH₃": latest["nh3"],
        "SO₂": latest["so2"],
        "CO": latest["co"],
        "O₃": latest["o3"]
    }
    
    col_pie1, col_pie2 = st.columns(2)
    
    with col_pie1:
        total_pollution = sum(pollutants_dict.values())
        contrib_pct = {k: (v/total_pollution)*100 for k, v in pollutants_dict.items()}
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(contrib_pct.keys()),
            values=list(contrib_pct.values()),
            marker=dict(colors=['#00e5ff','#00c853','#ffeb3b','#ff9800','#ff5722','#e91e63','#9c27b0']),
            hovertemplate="<b>%{label}</b><br>Contribution: %{value:.1f}%<extra></extra>"
        )])
        fig_pie.update_layout(template="plotly_dark", height=350, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_pie2:
        st.markdown("**Pollutant Levels & Health Impact:**")
        for pollutant, level in pollutants_dict.items():
            if pollutant == "PM2.5":
                impact = "⚠️ Penetrates lungs - respiratory diseases"
            elif pollutant == "PM10":
                impact = "⚠️ Irritates respiratory system"
            elif pollutant == "NO₂":
                impact = "⚠️ Causes airway inflammation"
            elif pollutant == "NH₃":
                impact = "⚠️ Eye, nose, throat irritation"
            elif pollutant == "SO₂":
                impact = "⚠️ Worsens asthma condition"
            elif pollutant == "CO":
                impact = "🔴 Reduces oxygen in blood"
            else:
                impact = "⚠️ Damages lung tissue"
            st.write(f"**{pollutant}:** {level:.1f} - {impact}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== UNIQUE FEATURE 2: ACTIVITY RECOMMENDATION ENGINE ==========
    st.subheader("🎯 Smart Activity Recommendations")
    
    if prediction <= 50:
        current = "Good (0-50)"
        acts = ["🏃 Running", "🚴 Cycling", "⛹️ Outdoor Sports"]
        time = "Unlimited"
        prot = "None needed"
        clr = "#00e676"
    elif prediction <= 100:
        current = "Moderate (51-100)"
        acts = ["🚶 Walking", "🧘 Light yoga"]
        time = "4+ hours"
        prot = "Optional mask"
        clr = "#ffd54f"
    elif prediction <= 200:
        current = "Unhealthy (101-200)"
        acts = ["🏋️ Indoor gym", "🎮 Gaming"]
        time = "30 min limit"
        prot = "N95 Mask"
        clr = "#ff9800"
    else:
        current = "Dangerous (201+)"
        acts = ["🏠 Stay indoors", "📺 Watch TV"]
        time = "Avoid outdoor"
        prot = "N95+ Indoors"
        clr = "#ff4b4b"
    
    st.markdown(f"""
    <div style="padding:20px; border-radius:15px; background:linear-gradient(135deg,{clr},#0e1117); border-left:6px solid {clr};">
    <h3>📌 Status: {current}</h3>
    <p><b>Safe Activities:</b> {', '.join(acts)}</p>
    <p><b>Time Limit:</b> {time}</p>
    <p><b>Protection:</b> {prot}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== UNIQUE FEATURE 3: AQI HISTORY COMPARISON ==========
    st.subheader("📈 Historical Comparison")
    
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    
    current_aqi = int(latest["aqi"])
    avg_7 = int(city_df.tail(7)["aqi"].mean())
    avg_30 = int(city_df.tail(30)["aqi"].mean())
    max_ev = int(city_df["aqi"].max())
    
    with col_h1:
        st.metric("Today's AQI", current_aqi)
    with col_h2:
        st.metric("Avg 7-Day", avg_7)
    with col_h3:
        st.metric("Avg 30-Day", avg_30)
    with col_h4:
        st.metric("Highest Ever", max_ev)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== UNIQUE FEATURE 4: MULTI-CITY COMPARISON ==========
    st.subheader("🏙️ Cities in {state} - AQI Comparison")
    
    state_cities = df[df["state"] == state].groupby("city").apply(lambda x: int(x.tail(1)["aqi"].values[0])).sort_values()
    
    if len(state_cities) > 0:
        fig_cmp = go.Figure(data=[go.Bar(
            y=state_cities.index,
            x=state_cities.values,
            orientation='h',
            marker=dict(
                color=state_cities.values,
                colorscale=[[0, '#00e676'], [0.5, '#ffd54f'], [1, '#ff4b4b']],
                colorbar=dict(title="AQI")
            )
        )])
        fig_cmp.update_layout(template="plotly_dark", height=350, xaxis_title="AQI", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig_cmp, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== UNIQUE FEATURE 5: WEATHER CORRELATION ==========
    st.subheader("🌡️ Weather & AQI Correlation")
    
    col_w1, col_w2, col_w3 = st.columns(3)
    
    with col_w1:
        st.metric("🌡️ Temperature", f"{latest['temperature']:.1f}°C")
    with col_w2:
        st.metric("💧 Humidity", f"{latest['humidity']:.1f}%")
    with col_w3:
        wind_status = "⚠️ Stagnant" if latest['humidity'] > 70 else "✅ Dispersing"
        st.write(f"**Air Movement:** {wind_status}")
    
    st.info("📌 **Weather Impact**: High humidity and low temperature trap pollutants, worsening AQI. Rain and wind help disperse pollution.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== UNIQUE FEATURE 6: PREDICTION CONFIDENCE ==========
    st.subheader("🎲 Prediction Confidence & Trend")
    
    recent = city_df.tail(7)["aqi"].values
    std_val = np.std(recent)
    conf_score = max(0, min(100, 100 - (std_val * 1.5)))
    trend = "📈 Improving ↓" if recent[-1] < recent[0] else "📉 Worsening ↑" if recent[-1] > recent[0] else "➡️ Stable"
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        fig_conf = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf_score,
            title="Confidence %",
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#00e676"}, 'steps': [
                {'range': [0, 50], 'color': "#ffebee"},
                {'range': [50, 100], 'color': "#e8f5e9"}
            ]},
            number={'suffix': "%"}
        ))
        fig_conf.update_layout(template="plotly_dark", height=300, paper_bgcolor="#0e1117")
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col_c2:
        st.markdown(f"""
        **📊 Model Confidence:**
        
        • **Score:** {conf_score:.1f}%
        • **Data Stability:** {'High ✅' if std_val < 20 else 'Moderate ⚠️' if std_val < 50 else 'Low 🔴'}
        • **Recent Trend:** {trend}
        • **Reliability:** {'Excellent 🟢' if conf_score > 80 else 'Good 🟡' if conf_score > 60 else 'Fair 🟠'}
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== UNIQUE FEATURE 7: HEALTH TIPS BY AQI LEVEL ==========
    with st.expander("💊 Health Recommendations by AQI Level"):
        st.markdown("""
        **🟢 Good (0-50):** No health concerns. Outdoor activities encouraged.
        
        **🟡 Moderate (51-100):** Sensitive groups should limit outdoor activities. Mask optional.
        
        **🟠 Unhealthy (101-200):**
        - Avoid prolonged outdoor exposure
        - Use N95 mask if going outside
        - Close windows, use air purifier
        - Seek medical help if symptoms appear
        
        **🔴 Very Unhealthy (201+):**
        - Stay indoors
        - Use N95/N100 masks if stepping out
        - Use HEPA air purifier
        - Seek immediate medical attention if difficulty breathing
        """)
    
    st.markdown("---")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- BEAUTIFUL POLLUTANT BREAKDOWN ----------
    st.subheader("🧪 Pollution Breakdown")

    pollutants = {
    "PM2.5": latest["pm25"],
    "PM10": latest["pm10"],
    "NO2": latest["no2"],
    "SO2": latest["so2"],
    "CO": latest["co"],
    "O3": latest["o3"]
}

    colors = ["#00e5ff","#00c853","#ffeb3b","#ff9800","#ff5722","#e91e63"]

    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=list(pollutants.keys()),
        y=list(pollutants.values()),
        text=[round(v,1) for v in pollutants.values()],
        textposition="outside",
        marker=dict(
        color=colors,
        line=dict(width=1, color="white")
    ),
    hovertemplate="<b>%{x}</b><br>Level: %{y}<extra></extra>"
))

    fig2.update_layout(
        template="plotly_dark",
        height=420,
        title="Major Pollutant Levels",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        xaxis_title="Pollutants",
        yaxis_title="Concentration",
        showlegend=False
)

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- BEAUTIFUL DOWNLOAD REPORT ----------
    st.subheader("📄 AQI Report")

    st.markdown(f"""
        <div style="
        padding:20px;
        border-radius:15px;
        background:linear-gradient(135deg,#0e1117,#161b22);
        border:1px solid #30363d;">
        <h3 style="color:#00e5ff;">📊 {city} Air Quality Summary</h3>
        <p><b>Date:</b> {datetime.now().strftime("%d %B %Y")}</p>
        <p><b>Predicted AQI:</b> {prediction}</p>
        <p><b>Status:</b> {status}</p>
        <p><b>Advice:</b> {advice}</p>
        </div>
""", unsafe_allow_html=True)

    report = f"""
    City: {city}
    Date: {datetime.now().strftime("%d-%m-%Y")}
    Predicted AQI: {prediction}
    Status: {status}

    Advice:
    {advice}
    """

    st.download_button(
        "⬇️ Download Detailed Report",
        report,
        "AQI_Report.txt",
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== MULTI-DAY FORECAST CALCULATION ==========
    forecast = [prediction]
    temp_features = features.copy()  # Shape: (1, 15)
    temp_aqi = prediction
    
    for day in range(1, days_to_forecast):
        # Update lag features for rolling forecast
        temp_features[0, 9] = temp_aqi  # aqi_lag1 (index 9 in feature array)
        temp_features[0, 12] = (forecast[0] + temp_aqi) / 2  # rolling_mean (index 12)
        
        # Predict next day
        next_pred = int(model.predict(temp_features)[0])
        forecast.append(max(0, next_pred))
        temp_aqi = next_pred
    
    # ========== UNIQUE FEATURE 8: AQI TREND PREDICTION ==========
    st.subheader("📈 AQI Trend Analysis")
    trend_col1, trend_col2, trend_col3 = st.columns(3)
    
    with trend_col1:
        day1 = forecast[0] if len(forecast) > 0 else prediction
        day7 = forecast[6] if len(forecast) > 6 else forecast[-1] if len(forecast) > 0 else prediction
        trend_direction = "📈 WORSENING" if day7 > day1 else "📉 IMPROVING" if day7 < day1 else "➡️ STABLE"
        trend_color = "#ff5722" if "WORSE" in trend_direction else "#00c853" if "IMPROV" in trend_direction else "#ffeb3b"
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0e1117,#161b22);padding:20px;border-radius:10px;border-left:5px solid {trend_color};">
        <b>7-Day Trend:</b> {trend_direction}
        <br><small>Day 1: {day1:.0f} → Day 7: {day7:.0f}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with trend_col2:
        forecast_max = max(forecast) if len(forecast) > 0 else prediction
        forecast_min = min(forecast) if len(forecast) > 0 else prediction
        volatility = forecast_max - forecast_min
        vol_status = "High 🌪️" if volatility > 30 else "Moderate 📊" if volatility > 15 else "Low 🌤️"
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0e1117,#161b22);padding:20px;border-radius:10px;border-left:5px solid #ff9800;">
        <b>Volatility Index:</b> {vol_status}
        <br><small>Range: {forecast_min:.0f} - {forecast_max:.0f} (Δ{volatility:.0f})</small>
        </div>
        """, unsafe_allow_html=True)
    
    with trend_col3:
        worst_day = np.argmax(forecast) + 1 if len(forecast) > 0 else 1
        worst_val = max(forecast) if len(forecast) > 0 else prediction
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0e1117,#161b22);padding:20px;border-radius:10px;border-left:5px solid #e91e63;">
        <b>Worst Day Alert:</b> Day {worst_day}
        <br><small>Peak AQI: {worst_val:.0f}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== UNIQUE FEATURE 9: POLLUTANT RISK ASSESSMENT ==========
    st.subheader("⚠️ Pollutant Health Risk Assessment")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        # WHO Guidelines Reference
        who_limits = {
            "PM2.5": {"limit": 15, "unit": "µg/m³"},
            "PM10": {"limit": 45, "unit": "µg/m³"},
            "NO2": {"limit": 40, "unit": "µg/m³", "key": "no2"},
            "O3": {"limit": 100, "unit": "µg/m³", "key": "o3"}
        }
        
        risk_data = []
        for pollutant, limits in who_limits.items():
            key = limits.get("key", pollutant.lower())
            actual = latest.get(key, 0) if isinstance(latest, dict) else getattr(latest, key, 0) if hasattr(latest, key) else latest[key] if key in latest.index else 0
            ratio = actual / limits["limit"] if limits["limit"] > 0 else 0
            risk_level = "🔴 Critical" if ratio > 2 else "🟠 High" if ratio > 1.5 else "🟡 Moderate" if ratio > 1 else "🟢 Safe"
            risk_data.append({"Pollutant": pollutant, "Level": f"{actual:.1f}", "WHO Limit": limits["limit"], "Ratio": f"{ratio:.2f}x", "Status": risk_level})
        
        risk_df = pd.DataFrame(risk_data)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    with risk_col2:
        # Health Impact Score
        health_score = 100
        if latest["pm25"] > 35: health_score -= 25
        elif latest["pm25"] > 15: health_score -= 15
        
        if latest["no2"] > 100: health_score -= 20
        elif latest["no2"] > 40: health_score -= 10
        
        if prediction > 150: health_score -= 20
        elif prediction > 100: health_score -= 10
        
        health_score = max(0, health_score)
        
        st.markdown(f"""
        <div style="text-align:center;padding:30px;background:linear-gradient(135deg,#0e1117,#161b22);border-radius:15px;">
        <h3 style="color:#00e5ff;">Overall Health Risk Score</h3>
        <div style="font-size:48px;font-weight:bold;color:{'#00c853' if health_score > 70 else '#ffeb3b' if health_score > 50 else '#ff9800' if health_score > 30 else '#ff5722'};">
        {health_score}%
        </div>
        <p style="font-size:14px;color:#b0b0b0;">{'Safe 🟢' if health_score > 70 else 'Moderate Risk 🟡' if health_score > 50 else 'High Risk 🟠' if health_score > 30 else 'Critical 🔴'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== UNIQUE FEATURE 10: SEASONAL COMPARISON ==========
    st.subheader("🌍 Seasonal & Historical Comparison")
    
    season_data = [
        {"Metric": "Current AQI", "Value": prediction},
        {"Metric": "7-Day Average", "Value": np.mean(forecast[:7]) if len(forecast) >= 7 else np.mean(forecast)},
        {"Metric": "Monthly Average", "Value": np.mean(forecast[:30]) if len(forecast) >= 30 else np.mean(forecast)},
        {"Metric": "All-Time High", "Value": df[df["city"] == city]["aqi"].max() if "aqi" in df.columns else prediction}
    ]
    
    season_df = pd.DataFrame(season_data)
    
    fig_season = go.Figure(data=[
        go.Bar(
            x=season_df["Metric"],
            y=season_df["Value"],
            text=[f"{v:.0f}" for v in season_df["Value"]],
            textposition="outside",
            marker=dict(
                color=["#ff5722" if v > 150 else "#ffeb3b" if v > 100 else "#00c853" for v in season_df["Value"]],
                line=dict(width=2, color="white")
            ),
            hovertemplate="<b>%{x}</b><br>AQI: %{y:.0f}<extra></extra>"
        )
    ])
    
    fig_season.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        showlegend=False,
        title="AQI Metrics Comparison"
    )
    
    st.plotly_chart(fig_season, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== UNIQUE FEATURE 11: OPTIMAL ACTIVITY WINDOW ==========
    st.subheader("🏃 Optimal Activity Planning")
    
    activity_col1, activity_col2, activity_col3 = st.columns(3)
    
    # Find best and worst days for outdoor activities
    best_day = np.argmin(forecast) + 1 if len(forecast) > 0 else 1
    best_aqi = min(forecast) if len(forecast) > 0 else prediction
    
    with activity_col1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#00c853,#66bb6a);padding:25px;border-radius:10px;text-align:center;">
        <h4 style="color:white;">✅ Best Day for Exercise</h4>
        <p style="font-size:28px;font-weight:bold;color:white;">Day {best_day}</p>
        <p style="font-size:14px;color:white;">AQI: {best_aqi:.0f} - Excellent for outdoor activities</p>
        </div>
        """, unsafe_allow_html=True)
    
    with activity_col2:
        avg_forecast_aqi = np.mean(forecast) if len(forecast) > 0 else prediction
        indoor_days = sum(1 for f in forecast if f > 100) if len(forecast) > 0 else 0
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#ffeb3b,#fdd835);padding:25px;border-radius:10px;text-align:center;color:#000;">
        <h4>⚠️ Indoor Days Recommended</h4>
        <p style="font-size:28px;font-weight:bold;">{indoor_days} Days</p>
        <p style="font-size:14px;">Out of {len(forecast)} days forecast</p>
        </div>
        """, unsafe_allow_html=True)
    
    with activity_col3:
        # Exercise guidelines based on AQI
        if prediction <= 50:
            exercise_rec = "All activities OK 🏋️"
            exercise_detail = "Full intensity exercises recommended"
        elif prediction <= 100:
            exercise_rec = "Light-Moderate 🚶"
            exercise_detail = "Limit intense exercise outdoors"
        elif prediction <= 200:
            exercise_rec = "Indoor Only 🏠"
            exercise_detail = "Exercise indoors today"
        else:
            exercise_rec = "Rest Day 😴"
            exercise_detail = "Minimize outdoor activities"
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#2196f3,#42a5f5);padding:25px;border-radius:10px;text-align:center;">
        <h4 style="color:white;">💪 Today's Exercise Level</h4>
        <p style="font-size:24px;font-weight:bold;color:white;">{exercise_rec}</p>
        <p style="font-size:13px;color:white;">{exercise_detail}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== UNIQUE FEATURE 12: EXCEEDANCE ANALYSIS ==========
    st.subheader("📊 Safety Threshold Exceedance Analysis")
    
    exceedance_col1, exceedance_col2, exceedance_col3, exceedance_col4 = st.columns(4)
    
    # Define thresholds
    good_threshold = 50
    moderate_threshold = 100
    unhealthy_threshold = 150
    hazardous_threshold = 200
    
    with exceedance_col1:
        exceed_good = sum(1 for f in forecast if f > good_threshold) if len(forecast) > 0 else 0
        pct_good = (exceed_good / len(forecast) * 100) if len(forecast) > 0 else 0
        
        st.metric(
            label="Days > Good Threshold (50)",
            value=f"{exceed_good}/{len(forecast) if len(forecast) > 0 else 1}",
            delta=f"{pct_good:.0f}%",
            delta_color="off"
        )
    
    with exceedance_col2:
        exceed_moderate = sum(1 for f in forecast if f > moderate_threshold) if len(forecast) > 0 else 0
        pct_moderate = (exceed_moderate / len(forecast) * 100) if len(forecast) > 0 else 0
        
        st.metric(
            label="Days > Moderate (100)",
            value=f"{exceed_moderate}/{len(forecast) if len(forecast) > 0 else 1}",
            delta=f"{pct_moderate:.0f}%",
            delta_color="inverse"
        )
    
    with exceedance_col3:
        exceed_unhealthy = sum(1 for f in forecast if f > unhealthy_threshold) if len(forecast) > 0 else 0
        pct_unhealthy = (exceed_unhealthy / len(forecast) * 100) if len(forecast) > 0 else 0
        
        st.metric(
            label="Days > Unhealthy (150)",
            value=f"{exceed_unhealthy}/{len(forecast) if len(forecast) > 0 else 1}",
            delta=f"{pct_unhealthy:.0f}%",
            delta_color="inverse"
        )
    
    with exceedance_col4:
        exceed_hazardous = sum(1 for f in forecast if f > hazardous_threshold) if len(forecast) > 0 else 0
        pct_hazardous = (exceed_hazardous / len(forecast) * 100) if len(forecast) > 0 else 0
        
        st.metric(
            label="Days > Hazardous (200)",
            value=f"{exceed_hazardous}/{len(forecast) if len(forecast) > 0 else 1}",
            delta=f"{pct_hazardous:.0f}%",
            delta_color="inverse"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== UNIQUE FEATURE 13: CUSTOM RISK ALERTS ==========
    st.subheader("🔔 Personalized Risk Alerts")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        age_group = st.multiselect(
            "Vulnerable Groups",
            ["Children", "Elderly", "Asthma", "Heart Disease", "Athletes"],
            default=["Asthma"]
        )
        
        risk_factors = {
            "Children": 1.3,
            "Elderly": 1.4,
            "Asthma": 1.5,
            "Heart Disease": 1.6,
            "Athletes": 0.8
        }
        
        max_risk = max([risk_factors.get(g, 1) for g in age_group]) if age_group else 1
        adjusted_aqi = prediction * max_risk
        
        if adjusted_aqi > 200:
            risk_alert = "🔴 CRITICAL - Seek medical attention if symptoms"
        elif adjusted_aqi > 150:
            risk_alert = "🟠 HIGH RISK - Minimize outdoor exposure"
        elif adjusted_aqi > 100:
            risk_alert = "🟡 MODERATE - Use N95 mask if outdoors"
        else:
            risk_alert = "🟢 LOW RISK - Proceed with normal activities"
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0e1117,#161b22);padding:20px;border-radius:10px;border-left:5px solid #00e5ff;">
        <b>Adjusted Risk Level:</b>
        <br>{risk_alert}
        <br><small>Adjusted AQI: {adjusted_aqi:.0f}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with alert_col2:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0e1117,#161b22);padding:20px;border-radius:10px;border-left:5px solid #ff9800;">
        <b>🏥 Health Precautions:</b>
        <ul style="font-size:13px;margin:10px 0;">
        <li>Use N95/KN95 mask when AQI > 100</li>
        <li>Keep inhalers and medications ready</li>
        <li>Install air purifiers in home</li>
        <li>Drink plenty of water</li>
        <li>Avoid vigorous outdoor exercise</li>
        <li>Keep windows closed during peaks</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ---------- BEAUTIFUL ALERT ----------
    if prediction > 200:
        st.error("🚨 HEALTH ALERT: Air quality is dangerous today!")
    elif prediction > 100:
        st.warning("⚠️ Moderate pollution detected. Take precautions.")
    else:
        st.success("✅ Air quality is safe today!")