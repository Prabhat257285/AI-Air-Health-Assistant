import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Air Health Assistant - India", layout="wide")

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
cities = sorted(df["city"].unique())

st.markdown("""
<style>

/* ===== Animated Purple–Black Background ===== */
.stApp {
    background: linear-gradient(
        -45deg,
        #0b0014,
        #1a0033,
        #2d004d,
        #000000
    );
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

/* Background Animation */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}


/* ===== Title ===== */
.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #ffffff;
    text-shadow: 0 0 25px #a855f7;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 22px;
    opacity: 0.85;
}

/* ===== Selectbox Fix (IMPORTANT) ===== */
div[data-baseweb="select"] > div {
    background-color: #12001f !important;
    border-radius: 12px;
    border: 1px solid #6d28d9;
}

div[data-baseweb="select"] span {
    color: white !important;
    font-size: 16px;
}

/* Dropdown options */
div[data-baseweb="menu"] {
    background-color: #12001f !important;
}

div[data-baseweb="menu"] span {
    color: white !important;
}

/* ===== Neon Button ===== */
.stButton>button {
    width: 100%;
    height: 65px;
    font-size: 22px;
    font-weight: bold;
    border-radius: 15px;
    border: none;
    background: linear-gradient(90deg,#9333ea,#22d3ee);
    color: black;
    box-shadow: 0 0 30px #9333ea;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 55px #9333ea;
}

/* Reduce top padding */
.block-container {
    padding-top: 2rem;
}
            
div.element-container,
div.stMarkdown,
div.stText,
div.stSelectbox,
div.stButton {
    background: transparent !important;
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="main-title">🌍 AI Air Health Assistant</div>
<div class="subtitle">Your Personal AI Doctor for the Air You Breathe</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- CITY SELECTION PANEL ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 2], gap="small")

with col1:
    state = st.selectbox("🗺️ Select State", states)

with col2:
    # Filter cities by selected state
    cities_in_state = sorted(df[df["state"] == state]["city"].unique())
    city = st.selectbox("🏙️ Select City", cities_in_state)

with col3:
    days_to_forecast = st.slider("📅 Days to Forecast", min_value=1, max_value=14, value=7)

st.markdown(
    "<div style='margin-top:10px; font-size:14px; opacity:0.9;'>"
    f"<b>{city}</b> • {state} | Forecast: Next {days_to_forecast} days"
    "</div>",
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREPARE DATA FOR MULTI-DAY FORECAST ----------
city_df = df[df["city"] == city].sort_values("date")

if len(city_df) < 4:
    st.error(f"Not enough historical data for {city}. Please select another city.")
    st.stop()

# Store predictions
predictions_list = []
forecast_dates = []

# Initial setup from last data point
last3 = city_df["aqi"].tail(3).values.astype(float)
latest = city_df.iloc[-1]
rolling_mean = city_df["aqi"].tail(7).mean()

# Get average values for future days (since we don't have future weather data)
avg_pm25 = city_df["pm25"].tail(7).mean()
avg_pm10 = city_df["pm10"].tail(7).mean()
avg_no2 = city_df["no2"].tail(7).mean()
avg_nh3 = city_df["nh3"].tail(7).mean()
avg_so2 = city_df["so2"].tail(7).mean()
avg_co = city_df["co"].tail(7).mean()
avg_o3 = city_df["o3"].tail(7).mean()
avg_temp = city_df["temperature"].tail(7).mean()
avg_humid = city_df["humidity"].tail(7).mean()

# Multi-day forecast loop
current_aqi_lag3 = last3[-3]
current_aqi_lag2 = last3[-2]
current_aqi_lag1 = last3[-1]
current_rolling_mean = rolling_mean

from datetime import timedelta

for day in range(days_to_forecast):
    # Create features for prediction
    features = np.array([[ 
        avg_pm25,
        avg_pm10,
        avg_no2,
        avg_nh3,              
        avg_so2,
        avg_co,
        avg_o3,
        avg_temp,
        avg_humid,
        current_aqi_lag1,
        current_aqi_lag2,
        current_aqi_lag3,
        current_rolling_mean,
        1 if city=="Jaipur" else 0,
        1 if city=="Lucknow" else 0
    ]])
    
    # Predict AQI for this day
    predicted_aqi = int(model.predict(features)[0])
    predictions_list.append(predicted_aqi)
    
    # Calculate forecast date
    forecast_date = datetime.now() + timedelta(days=day+1)
    forecast_dates.append(forecast_date.strftime("%d-%b-%Y"))
    
    # Update lag features for next iteration
    current_aqi_lag3 = current_aqi_lag2
    current_aqi_lag2 = current_aqi_lag1
    current_aqi_lag1 = predicted_aqi
    current_rolling_mean = (current_aqi_lag1 + current_aqi_lag2 + current_aqi_lag3 + rolling_mean * 4) / 7

st.markdown("<br>", unsafe_allow_html=True)


# ---------- AI PREDICTION BUTTON ----------
predictions = None

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.markdown("### 🤖 AI Prediction Engine")

if st.button(f"🚀 Predict Next {days_to_forecast} Days AQI"):
    with st.spinner(f"AI is analyzing air patterns for the next {days_to_forecast} days..."):
        time.sleep(2)
        predictions = predictions_list

st.markdown('</div>', unsafe_allow_html=True)


# ---------- ONLY SHOW RESULTS AFTER CLICK ----------
if predictions:

    # ---------- AQI CATEGORY ----------
    def aqi_info(aqi):
        if aqi <= 50: return "GOOD 😊", "#00e676"
        elif aqi <= 100: return "MODERATE 🙂", "#ffd54f"
        elif aqi <= 200: return "UNHEALTHY ⚠️", "#ff9800"
        else: return "VERY DANGEROUS 🚨", "#ff4b4b"

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- CREATE TABS FOR DIFFERENT VIEWS ----------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Forecast", 
        "🔄 Multi-City Compare", 
        "🧪 Pollution Analysis",
        "🎯 Activity Guide",
        "📈 Regional Stats"
    ])

    with tab1:
        st.subheader("📅 AQI Forecast for Next Days")
    
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Predicted AQI": predictions_list,
        "Status": [aqi_info(aqi)[0] for aqi in predictions_list]
    })
    
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- FORECAST CHART ----------
    st.subheader("📊 AQI Forecast Trend")
    
    fig_forecast = go.Figure()
    
    fig_forecast.add_trace(go.Scatter(
        x=forecast_dates,
        y=predictions_list,
        mode="lines+markers",
        line=dict(width=4, color="#00e5ff"),
        marker=dict(size=10, color="#00e5ff"),
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.12)",
        hovertemplate="<b>Date:</b> %{x}<br><b>AQI:</b> %{y}<extra></extra>"
    ))
    
    # Add reference lines for AQI categories
    fig_forecast.add_hline(y=50, line_dash="dash", line_color="#00e676", annotation_text="Good", annotation_position="right")
    fig_forecast.add_hline(y=100, line_dash="dash", line_color="#ffd54f", annotation_text="Moderate", annotation_position="right")
    fig_forecast.add_hline(y=200, line_dash="dash", line_color="#ff9800", annotation_text="Unhealthy", annotation_position="right")
    
    fig_forecast.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(
            text=f"Air Quality Forecast for {city}",
            font=dict(size=22)
        ),
        xaxis=dict(
            showgrid=False,
            title="Date"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            title="AQI Level",
            range=[0, max(300, max(predictions_list) + 50)]
        ),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- TODAY'S PREDICTION HIGHLIGHTS ----------
    st.subheader("🎯 Key Insights")
    
    prediction_today = predictions_list[0]
    status_today, color_today = aqi_info(prediction_today)
    max_aqi = max(predictions_list)
    min_aqi = min(predictions_list)
    avg_aqi = int(np.mean(predictions_list))
    max_day_idx = predictions_list.index(max_aqi)
    min_day_idx = predictions_list.index(min_aqi)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tomorrow's AQI", prediction_today, delta=f"Status: {status_today}")
    
    with col2:
        st.metric("Max AQI", max_aqi, delta=f"on {forecast_dates[max_day_idx]}")
    
    with col3:
        st.metric("Min AQI", min_aqi, delta=f"on {forecast_dates[min_day_idx]}")
    
    with col4:
        st.metric("Average AQI", avg_aqi, delta="Next 7 days")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- BEAUTIFUL HEALTH IMPACT ----------
    st.subheader("👨‍👩‍👧‍👦 Health Risk Analysis (Tomorrow)")

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
              "HIGH" if prediction_today > 100 else "LOW",
              "#ff4b4b" if prediction_today > 100 else "#00e676")

    with c2:
        risk_card("👶 Children",
              "MODERATE" if prediction_today > 80 else "LOW",
              "#ff9800" if prediction_today > 80 else "#00e676")

    with c3:
        risk_card("👴 Elderly",
              "HIGH" if prediction_today > 120 else "LOW",
              "#ff4b4b" if prediction_today > 120 else "#00e676")

    with c4:
        risk_card("🏃 Healthy Adults",
              "MODERATE" if prediction_today > 150 else "LOW",
              "#ff9800" if prediction_today > 150 else "#00e676")

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------AI ADVICE CARD----------
    st.subheader("🤖 AI Health Advice (Tomorrow)")

    if prediction_today < 100:
        advice = "🌿 Great day! Enjoy outdoor activities safely."
        bg = "#0f5132"
        border = "#00ff9d"

    elif prediction_today < 200:
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
    st.subheader("📊 Historical AQI + Forecast")

    past = city_df.tail(30)
    latest = city_df.iloc[-1]

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=past["date"],
        y=past["aqi"],
        mode="lines+markers",
        line=dict(width=4, color="#00e5ff"),
        marker=dict(size=8, color="#00e5ff"),
        fill="tozeroy",
        fillcolor="rgba(0,229,255,0.12)",
        name="Historical AQI",
        hovertemplate="<b>Date:</b> %{x}<br><b>AQI:</b> %{y}<extra></extra>"
    ))

    # Forecast data
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predictions_list,
        mode="lines+markers",
        line=dict(width=4, color="#ff6b6b", dash="dash"),
        marker=dict(size=8, color="#ff6b6b"),
        fill="tozeroy",
        fillcolor="rgba(255,107,107,0.12)",
        name="Forecast AQI",
        hovertemplate="<b>Date:</b> %{x}<br><b>AQI:</b> %{y}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(
            text=f"Historical Data & Forecast - {city}",
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
        hovermode="x unified",
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0.5)")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- BEAUTIFUL POLLUTANT BREAKDOWN ----------
    st.subheader("🧪 Current Pollution Breakdown")

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
        title="Major Pollutant Levels (Latest)",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        xaxis_title="Pollutants",
        yaxis_title="Concentration",
        showlegend=False
)

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- BEAUTIFUL DOWNLOAD REPORT ----------
    st.subheader("📄 AQI Forecast Report")

    report_lines = [
        f"City: {city}",
        f"Report Generated: {datetime.now().strftime('%d %B %Y')}",
        f"Forecast Days: {days_to_forecast}",
        "",
        "=" * 50,
        "AQI FORECAST DETAILS",
        "=" * 50,
        ""
    ]
    
    for i, (date, aqi) in enumerate(zip(forecast_dates, predictions_list)):
        status, _ = aqi_info(aqi)
        report_lines.append(f"Day {i+1} ({date}): AQI = {aqi} | Status: {status}")
    
    report_lines.extend([
        "",
        "=" * 50,
        "SUMMARY",
        "=" * 50,
        f"Tomorrow's AQI: {prediction_today}",
        f"Maximum AQI: {max_aqi} (on {forecast_dates[max_day_idx]})",
        f"Minimum AQI: {min_aqi} (on {forecast_dates[min_day_idx]})",
        f"Average AQI: {avg_aqi}",
        "",
        f"Tomorrow's Advice: {advice}",
    ])
    
    report = "\n".join(report_lines)

    st.markdown(f"""
        <div style="
        padding:20px;
        border-radius:15px;
        background:linear-gradient(135deg,#0e1117,#161b22);
        border:1px solid #30363d;">
        <h3 style="color:#00e5ff;">📊 {city} AQI Forecast Report</h3>
        <p><b>Generated:</b> {datetime.now().strftime("%d %B %Y")}</p>
        <p><b>Forecast Period:</b> Next {days_to_forecast} days</p>
        <p><b>Tomorrow's AQI:</b> {prediction_today}</p>
        <p><b>Forecast Range:</b> {min_aqi} - {max_aqi}</p>
        </div>
""", unsafe_allow_html=True)

    st.download_button(
    "⬇️ Download Complete Report",
    report,
    f"AQI_Forecast_{city}_{datetime.now().strftime('%Y-%m-%d')}.txt",
    use_container_width=True
)

    st.markdown("<br>", unsafe_allow_html=True)
    # ---------- BEAUTIFUL ALERT ----------
    if max_aqi > 200:
        st.error(f"🚨 HEALTH ALERT: Air quality will be dangerous! Maximum AQI expected: {max_aqi}")
    elif max_aqi > 100:
        st.warning(f"⚠️ Moderate to unhealthy pollution expected. Max AQI: {max_aqi}")
    else:
        st.success(f"✅ Air quality will be mostly safe in next {days_to_forecast} days!")

# ---------- TAB 2: MULTI-CITY COMPARISON ----------
with tab2:
    st.subheader("🌍 Compare Multiple Cities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        compare_state = st.selectbox("Select State for Comparison", states, key="compare_state")
    
    with col2:
        compare_cities_opts = sorted(df[df["state"] == compare_state]["city"].unique())
        selected_cities = st.multiselect(
            "Select Cities to Compare", 
            compare_cities_opts,
            default=compare_cities_opts[:3] if len(compare_cities_opts) >= 3 else compare_cities_opts
        )
    
    if selected_cities:
        # Create comparison data
        comparison_data = []
        for comp_city in selected_cities:
            comp_df = df[df["city"] == comp_city].sort_values("date")
            if len(comp_df) > 0:
                latest_comp = comp_df.iloc[-1]
                avg_aqi_comp = comp_df["aqi"].tail(30).mean()
                comparison_data.append({
                    "City": comp_city,
                    "Current AQI": int(latest_comp["aqi"]),
                    "Avg AQI (30d)": int(avg_aqi_comp),
                    "PM2.5": round(latest_comp["pm25"], 1),
                    "PM10": round(latest_comp["pm10"], 1),
                    "NO₂": round(latest_comp["no2"], 1)
                })
        
        comp_df_display = pd.DataFrame(comparison_data)
        st.dataframe(comp_df_display, use_container_width=True, hide_index=True)
        
        # Comparison chart
        fig_compare = go.Figure()
        
        for comp_city in selected_cities:
            comp_hist = df[df["city"] == comp_city].sort_values("date").tail(30)
            fig_compare.add_trace(go.Scatter(
                x=comp_hist["date"],
                y=comp_hist["aqi"],
                mode="lines+markers",
                name=comp_city,
                hovertemplate="<b>%{fullData.name}</b><br>AQI: %{y}<extra></extra>"
            ))
        
        fig_compare.update_layout(
            template="plotly_dark",
            height=400,
            title="AQI Comparison - Last 30 Days",
            xaxis_title="Date",
            yaxis_title="AQI",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)

# ---------- TAB 3: POLLUTION ANALYSIS ----------
with tab3:
    st.subheader("🧪 Pollution Contribution Analysis")
    
    latest_vals = city_df.iloc[-1]
    pollutant_levels = {
        "PM2.5": latest_vals["pm25"],
        "PM10": latest_vals["pm10"],
        "NO₂": latest_vals["no2"],
        "NH₃": latest_vals["nh3"],
        "SO₂": latest_vals["so2"],
        "CO": latest_vals["co"],
        "O₃": latest_vals["o3"]
    }
    
    # Calculate contribution percentage
    total_pollution = sum(pollutant_levels.values())
    contributions = {k: (v/total_pollution)*100 for k, v in pollutant_levels.items()}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Pollutant Concentration Levels:**")
        for pollutant, level in pollutant_levels.items():
            st.metric(pollutant, f"{level:.1f} µg/m³")
    
    with col2:
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(contributions.keys()),
            values=list(contributions.values()),
            marker=dict(colors=["#00e5ff","#00c853","#ffeb3b","#ff9800","#ff5722","#e91e63","#9c27b0"]),
            hovertemplate="<b>%{label}</b><br>Contribution: %{value:.1f}%<extra></extra>"
        )])
        
        fig_pie.update_layout(
            template="plotly_dark",
            height=400,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117"
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Health impact of each pollutant
    st.markdown("**Health Impact by Pollutant:**")
    
    pollutant_effects = {
        "PM2.5": "⚠️ Penetrates deep into lungs, causes respiratory & heart issues",
        "PM10": "⚠️ Irritates respiratory system, reduces visibility",
        "NO₂": "⚠️ Causes inflammation of airways, asthma attacks",
        "NH₃": "⚠️ Eye, nose, throat irritation",
        "SO₂": "⚠️ Affects respiratory system, worsens asthma",
        "CO": "🔴 Reduces oxygen in blood, affects heart function",
        "O₃": "⚠️ Damages lung tissue, reduces lung capacity"
    }
    
    for pollutant, effect in pollutant_effects.items():
        st.write(f"**{pollutant}**: {effect}")

# ---------- TAB 4: ACTIVITY RECOMMENDATIONS ----------
with tab4:
    st.subheader("🎯 Safe Activity Guide")
    
    # Use today's forecast AQI
    aqi_level = predictions_list[0] if predictions_list else latest_vals["aqi"]
    
    activity_recommendations = {
        "Good (0-50)": {
            "Safe Activities": ["🏃 Running", "🚴 Cycling", "⛹️ Outdoor Sports", "🧘 Yoga", "🚶 Walking"],
            "Outdoor Time": "✅ Unlimited",
            "Protection": "None needed",
            "Color": "#00e676"
        },
        "Moderate (51-100)": {
            "Safe Activities": ["🚶 Walking", "🧘 Light Exercise", "🛴 Skating", "🏌️ Golf"],
            "Outdoor Time": "✅ Up to 4 hours",
            "Protection": "Consider mask for sensitive groups",
            "Color": "#ffd54f"
        },
        "Unhealthy (101-200)": {
            "Safe Activities": ["🧘 Indoor Exercise", "🏋️ Gym", "🎮 Indoor Recreation"],
            "Outdoor Time": "⚠️ Limit to 30 mins",
            "Protection": "N95 Mask Recommended",
            "Color": "#ff9800"
        },
        "Very Unhealthy (201+)": {
            "Safe Activities": ["🏋️ Gym", "🎮 Gaming", "📚 Reading", "🧩 Puzzles"],
            "Outdoor Time": "🚫 Avoid",
            "Protection": "N95+ Mask + Stay Indoors",
            "Color": "#ff4b4b"
        }
    }
    
    # Determine current level
    if aqi_level <= 50:
        current_level = "Good (0-50)"
    elif aqi_level <= 100:
        current_level = "Moderate (51-100)"
    elif aqi_level <= 200:
        current_level = "Unhealthy (101-200)"
    else:
        current_level = "Very Unhealthy (201+)"
    
    current_rec = activity_recommendations[current_level]
    
    st.markdown(f"""
    <div style="
    padding:20px;
    border-radius:15px;
    background:linear-gradient(135deg,{current_rec['Color']},#0e1117);
    border-left:6px solid {current_rec['Color']};
    ">
    <h3>Current Status: {current_level}</h3>
    <h4>✅ Safe Activities:</h4>
    <p>{', '.join(current_rec['Safe Activities'])}</p>
    <h4>⏰ Recommended Outdoor Time:</h4>
    <p>{current_rec['Outdoor Time']}</p>
    <h4>🛡️ Protection Needed:</h4>
    <p>{current_rec['Protection']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Activity Guidelines for All Levels:**")
    for level, rec in activity_recommendations.items():
        with st.expander(f"📌 {level}"):
            st.write(f"**Activities:** {', '.join(rec['Safe Activities'])}")
            st.write(f"**Time Limit:** {rec['Outdoor Time']}")
            st.write(f"**Protection:** {rec['Protection']}")

# ---------- TAB 5: REGIONAL STATISTICS ----------
with tab5:
    st.subheader("📈 Regional Air Quality Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Best & Worst Cities in Selected State**")
        
        state_data = df[df["state"] == state].copy()
        latest_by_city = state_data.sort_values("date").groupby("city").tail(1)
        
        best_cities = latest_by_city.nsmallest(3, "aqi")[["city", "aqi"]]
        worst_cities = latest_by_city.nlargest(3, "aqi")[["city", "aqi"]]
        
        st.markdown("🟢 **Best Air Quality (Cleanest):**")
        for idx, row in best_cities.iterrows():
            st.write(f"• {row['city']}: AQI {int(row['aqi'])}")
        
        st.markdown("🔴 **Worst Air Quality (Most Polluted):**")
        for idx, row in worst_cities.iterrows():
            st.write(f"• {row['city']}: AQI {int(row['aqi'])}")
    
    with col2:
        st.markdown("**All-India State Rankings**")
        
        # Get latest AQI for all cities
        national_data = df.sort_values("date").groupby("state").apply(lambda x: x.nlargest(1, "date")).reset_index(drop=True)
        state_avg_aqi = national_data.groupby("state")["aqi"].mean().sort_values(ascending=False)
        
        st.markdown("📊 **Average AQI by State:**")
        
        fig_state = go.Figure(data=[
            go.Bar(
                x=state_avg_aqi.index,
                y=state_avg_aqi.values,
                marker=dict(
                    color=state_avg_aqi.values,
                    colorscale=[[0, "#00e676"], [0.5, "#ffd54f"], [1, "#ff4b4b"]],
                    colorbar=dict(title="AQI")
                ),
                hovertemplate="<b>%{x}</b><br>Avg AQI: %{y:.0f}<extra></extra>"
            )
        ])
        
        fig_state.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="State",
            yaxis_title="Average AQI",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117"
        )
        
        st.plotly_chart(fig_state, use_container_width=True, key="state_chart")