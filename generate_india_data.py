import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# All major cities from Indian states and union territories
cities_states = {
    # States
    "Delhi": "Delhi",
    "Mumbai": "Maharashtra",
    "Bangalore": "Karnataka",
    "Hyderabad": "Telangana",
    "Chennai": "Tamil Nadu",
    "Kolkata": "West Bengal",
    "Pune": "Maharashtra",
    "Ahmedabad": "Gujarat",
    "Jaipur": "Rajasthan",
    "Lucknow": "Uttar Pradesh",
    "Chandigarh": "Punjab",
    "Indore": "Madhya Pradesh",
    "Bhopal": "Madhya Pradesh",
    "Surat": "Gujarat",
    "Vadodara": "Gujarat",
    "Guwahati": "Assam",
    "Visakhapatnam": "Andhra Pradesh",
    "Kochi": "Kerala",
    "Trivandrum": "Kerala",
    "Coimbatore": "Tamil Nadu",
    "Nagpur": "Maharashtra",
    "Srinagar": "Jammu & Kashmir",
    "Ludhiana": "Punjab",
    "Amritsar": "Punjab",
    "Ranchi": "Jharkhand",
    "Patna": "Bihar",
    "Varanasi": "Uttar Pradesh",
    "Meerut": "Uttar Pradesh",
    "Noida": "Uttar Pradesh",
    "Ghaziabad": "Uttar Pradesh",
    "Faridabad": "Haryana",
    "Panipat": "Haryana",
    "Hisar": "Haryana",
    "Raipur": "Chhattisgarh",
    "Bilaspur": "Chhattisgarh",
    "Vadodara": "Gujarat",
    "Rajkot": "Gujarat",
    "Thiruvananthapuram": "Kerala",
    "Ernakulam": "Kerala",
    "Agra": "Uttar Pradesh",
    "Indore": "Madhya Pradesh",
    "Jabalpur": "Madhya Pradesh",
    "Gwalior": "Madhya Pradesh",
}

# AQI variation by region (base AQI multipliers)
region_aqi_factors = {
    "Uttar Pradesh": 1.2,
    "Delhi": 1.3,
    "Haryana": 1.15,
    "Punjab": 1.1,
    "Maharashtra": 0.95,
    "Gujarat": 0.9,
    "Karnataka": 0.85,
    "Tamil Nadu": 0.8,
    "Telangana": 0.75,
    "West Bengal": 0.95,
    "Rajasthan": 0.9,
    "Madhya Pradesh": 1.0,
    "Andhra Pradesh": 0.8,
    "Assam": 0.85,
    "Kerala": 0.7,
    "Jammu & Kashmir": 0.9,
    "Jharkhand": 1.1,
    "Bihar": 1.05,
    "Chhattisgarh": 1.0,
}

# Generate data
start_date = datetime(2025, 1, 1)
dates = [start_date + timedelta(days=7*i) for i in range(52)]

data = []

for cities_list in [list(cities_states.keys())]:
    for city in cities_list:
        state = cities_states[city]
        base_factor = region_aqi_factors.get(state, 1.0)
        
        for i, date in enumerate(dates):
            # Seasonal variation
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
            
            # Random noise
            noise = np.random.normal(1, 0.1)
            
            # Calculate AQI components
            pm25_base = 80 * base_factor * seasonal_factor * noise
            pm10_base = 120 * base_factor * seasonal_factor * noise
            no2_base = 50 * base_factor * noise
            
            # Temperature and humidity vary by season
            temp_base = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            humidity_base = 60 + 15 * np.cos(2 * np.pi * (day_of_year - 80) / 365)
            
            # Temperature and region-specific variations
            if state in ["Maharashtra", "Karnataka", "Tamil Nadu", "Telangana", "Andhra Pradesh"]:
                temp_base += 5
            elif state in ["Jammu & Kashmir", "Punjab", "Haryana"]:
                temp_base -= 3
            elif state == "Kerala":
                humidity_base += 10
                temp_base += 3
            
            # Other pollutants
            nh3 = max(2, np.random.normal(8, 3) * base_factor)
            so2 = max(2, np.random.normal(15, 5) * base_factor)
            co = max(5, np.random.normal(50, 20) * base_factor)
            o3 = max(5, np.random.normal(25, 10) * (1 / base_factor))
            
            # Calculate AQI based on pollutants
            aqi = max(pm25_base, pm10_base, no2_base, nh3*5, so2*5, co/2, o3*2)
            aqi = int(aqi)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'city': city,
                'state': state,
                'pm25': round(pm25_base, 1),
                'pm10': round(pm10_base, 1),
                'no2': round(no2_base, 1),
                'nh3': round(nh3, 1),
                'so2': round(so2, 1),
                'co': round(co, 1),
                'o3': round(o3, 1),
                'temperature': round(temp_base, 1),
                'humidity': round(humidity_base, 1),
                'aqi': aqi
            })

# Create DataFrame
df = pd.DataFrame(data)

# Remove state column before saving (keep original format)
df_save = df[['date', 'city', 'pm25', 'pm10', 'no2', 'nh3', 'so2', 'co', 'o3', 'temperature', 'humidity', 'aqi']]
df_save = df_save.sort_values(['city', 'date'])

# Save to CSV
df_save.to_csv('air_quality.csv', index=False)

# Also save with state info for reference
df_with_state = df[['date', 'city', 'state', 'pm25', 'pm10', 'no2', 'nh3', 'so2', 'co', 'o3', 'temperature', 'humidity', 'aqi']]
df_with_state = df_with_state.sort_values(['state', 'city', 'date'])
df_with_state.to_csv('air_quality_with_state.csv', index=False)

print("✓ Data generated successfully!")
print(f"Total records: {len(df_save)}")
print(f"Unique cities: {df['city'].nunique()}")
print(f"Unique states: {df['state'].nunique()}")
print(f"\nSample data:")
print(df_with_state.head(10))
