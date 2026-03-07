# AI Air Health Assistant 💨

[![Python](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.28-orange?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/Varun-men/AI-Air-Health-Assistant?style=for-the-badge)](https://github.com/Varun-men/AI-Air-Health-Assistant)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

# 🌍 AI Air Health Assistant

An end-to-end Machine Learning project that predicts **Air Quality Index (AQI)** based on pollutant concentrations and provides **health recommendations** through an interactive web dashboard.

The system analyzes air pollution data, trains a machine learning model, and deploys it as a real-time prediction application.

---

#  Live Application

🔗 **Streamlit App:**
https://ai-air-health-assistant-2yvc4akzjumhthvaw8usbg.streamlit.app/

🔗 **GitHub Repository:**
https://github.com/Varun-men/AI-Air-Health-Assistant

---

#  Project Overview

Air pollution is a major environmental and public health concern.
This project builds an **AI-powered system** that predicts AQI levels based on pollutant concentrations and provides actionable health insights.

The project includes:

* Data analysis and visualization
* Machine learning model development
* Feature engineering
* Model deployment
* Interactive dashboard

---

#  Machine Learning Pipeline

Dataset
↓
Data Preprocessing
↓
Feature Engineering
↓
Model Training & Evaluation
↓
Model Saving
↓
Streamlit Web Application
↓
Real-time AQI Prediction

---

#  Features

* Real-time **AQI prediction**
* **City-wise AQI analysis**
* **Color-coded AQI severity indicator**
* **AI-generated health recommendations**
* **365-day AQI trend visualization**
* **Pollutant concentration charts**
* **Health risk metrics**
* **Downloadable AQI report**
* **Alert notifications for hazardous AQI**

---

# Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Joblib

### Visualization

* Plotly
* Matplotlib

### Deployment

* Streamlit

### Version Control

* Git
* GitHub

---

#  Project Structure

```
AI-Air-Health-Assistant
│
├── app.py
├── requirements.txt
├── best_aqi_model.pkl
├── feature_columns.pkl
│
├── data
│   └── air_quality_dataset.csv
│
├── notebooks
│   └── EDA_and_Model_Training.ipynb
│
├── utils
│   └── helper_functions.py
│
└── README.md
```

---

#  Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/AI-Air-Health-Assistant.git
```

### 2️⃣ Navigate to project folder

```
cd AI-Air-Health-Assistant
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app

```
streamlit run app.py
```

---

#  Model Details

**Problem Type:** Regression

The model predicts AQI using pollutant concentrations such as:

* PM2.5
* PM10
* NO₂
* SO₂
* CO
* O₃

### Model Output

Predicted AQI value along with severity category and health recommendations.

---

#  Model Files

```
best_aqi_model.pkl
```

Trained machine learning model.

```
feature_columns.pkl
```

Stores feature order used during training to ensure consistent prediction input.

---

#  Dashboard Preview

Example:


<img width="1882" height="855" alt="Screenshot 2026-03-07 134328" src="https://github.com/user-attachments/assets/415476be-144c-4919-9c83-b51b56aef030" />



---

#  Key Insights from Data

* AQI varies significantly across cities.
* Particulate matter (PM2.5 and PM10) are dominant pollution contributors.
* Pollution shows strong seasonal trends.
* Outliers correspond to real-world pollution spikes.

---

#  Limitations

* Uses historical pollution data.
* Does not include real-time sensor data.
* Accuracy depends on dataset quality.

---

#  Future Improvements

* Real-time AQI API integration
* Mobile responsive interface
* Geolocation-based pollution alerts
* Deep learning model comparison

---

#  References

* Central Pollution Control Board (CPCB)
* World Health Organization (WHO)
* Scikit-learn Documentation
* Streamlit Documentation
* Plotly Documentation

---

#  Author

**Varun Kumar**

Machine Learning & Data Science Project


Passionate about AI, data science, and creating interactive web apps.
