# AI Air Health Assistant 💨

[![Python](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.28-orange?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/Varun-men/AI-Air-Health-Assistant?style=for-the-badge)](https://github.com/Varun-men/AI-Air-Health-Assistant)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## Project Overview

**AI Air Health Assistant** is a premium **Streamlit web application** that predicts **Air Quality Index (AQI)** for any city and provides **real-time health risk alerts**. It uses a trained **RandomForestRegressor ML model**, combines historical AQI trends, pollutant levels, and weather data, and presents results in a visually engaging dashboard.

![App Demo GIF](assets/ai_air_health_demo.gif)
---

## Features ✨

- Predict AQI dynamically for selected cities  
- Health advice based on AQI severity (Good, Moderate, Unhealthy, Hazardous)  
- 30-day AQI trend visualization using **Plotly**  
- Pollutant breakdown bar charts (PM2.5, PM10, NO₂, SO₂, CO, O₃)  
- Rolling mean and last 3 AQI values as predictive features  
- Premium Streamlit UI:
  - Hero AQI section with severity meter  
  - Downloadable AQI reports  
  - Alert notifications for dangerous AQI  
  - Glassmorphism cards and neon buttons  

---

## Project Structure


AI-Air-Health-Assistant/
│
├── app.py # Streamlit app
├── best_aqi_model.pkl # Trained RandomForestRegressor
├── air_quality.csv # Dataset (if needed for retraining)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── utils.py # Helper functions (optional)
└── assets/ # Images, icons, GIFs, CSS


---

## Installation & Setup ⚡

1. **Clone the repository**

```bash
git clone https://github.com/Varun-men/AI-Air-Health-Assistant.git
cd AI-Air-Health-Assistant
```


2. **Create and activate a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash

pip install -r requirements.txt
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```
Open the URL displayed in your terminal to explore the app.


## How It Works 🔍

1. Data Loading: Loads historical air quality data (air_quality.csv)
2. Preprocessing: Cleans missing values, computes rolling means, prepares features
3. Feature Engineering: Combines pollutant levels, weather data, last 3 AQI values, and city one-hot encoding
4. Model Prediction: Uses RandomForestRegressor to predict next-day AQI
5. UI Display: Shows AQI, severity meter, health advice, 30-day trend chart, and pollutant breakdown
6. Alerts: Provides warning notifications for hazardous AQI

## Screenshots & GIFs 🎬

<img width="1857" height="850" alt="image" src="https://github.com/user-attachments/assets/c07821db-0cad-4ea6-b3b2-882c3677b981" />


## Dependencies 🛠️

1. Python 3.11+
2. pandas, numpy
3. scikit-learn
4. streamlit
5. plotly
6. joblib

Install all dependencies via:
```bash
pip install -r requirements.txt
```

## License 📄

This project is licensed under the MIT License.

## Author 👨‍💻

Varun Kumar

GitHub: @Varun-men

Passionate about AI, data science, and creating interactive web apps.
# AI-Air-Health-Assistant

c7d0223 (Initial commit)
