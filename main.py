import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Generate Synthetic Data
np.random.seed(42)
n_samples = 1000
spo2 = np.random.normal(95, 5, n_samples).clip(70, 100)
pulse = np.random.normal(75, 15, n_samples).clip(40, 120)
status = np.where((spo2 < 90) | (pulse < 60) | (pulse > 100), 1, 0)
data = pd.DataFrame({'SpO2': spo2, 'PulseRate': pulse, 'Status': status})

# Step 2: Data Preprocessing
scaler = StandardScaler()
X = data[['SpO2', 'PulseRate']]
y = data['Status']
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train Model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Step 4: Streamlit UI
st.title("Pulse Oximeter Remote Patient Monitoring")
st.write(f"Model Accuracy: {accuracy:.2f}")

# User Input
spo2_input = st.number_input("Enter SpO2 Level (70-100):", min_value=70, max_value=100, value=95)
pulse_input = st.number_input("Enter Pulse Rate (40-120):", min_value=40, max_value=120, value=75)

# Prediction Button
if st.button("Predict Health Status"):
    input_data = np.array([[spo2_input, pulse_input]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    status_text = "Abnormal (Needs Attention)" if prediction[0] == 1 else "Normal (Healthy)"
    st.success(f"Predicted Health Status: {status_text}")
