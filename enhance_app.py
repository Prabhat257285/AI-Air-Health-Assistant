#!/usr/bin/env python3
"""
Enhanced AI Air Health Assistant - Complete Version
Adds unique features to the application
"""

# Read the clean app
with open('app_clean.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "df = pd.read_csv("air_quality.csv")"  
# and replace with enhanced data loading
for i, line in enumerate(lines):
    if 'df = pd.read_csv("air_quality.csv")' in line:
        lines[i] = '''try:
    df = pd.read_csv("air_quality_with_state.csv")
except FileNotFoundError:
    df = pd.read_csv("air_quality.csv")
    if "state" not in df.columns:
        df["state"] = "Unknown"

states = sorted(df["state"].unique())
'''
        break

# Find "cities = sorted(df["city"].unique())" and update it
for i, line in enumerate(lines):
    if 'cities = sorted(df["city"].unique())' in line:
        # Remove this line since we'll handle cities dynamically now
        lines[i] = ''
        break

# Find where to add timedelta import
for i, line in enumerate(lines):
    if 'from datetime import datetime' in line:
        lines[i] = 'from datetime import datetime, timedelta\n'
        break

# Write the modified file
with open('app_enhanced.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✓ Created app_enhanced.py")
print("Next: Manually verify and use this as base")
