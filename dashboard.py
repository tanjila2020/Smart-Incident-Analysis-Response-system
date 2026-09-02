"""
dashboard.py  --  Streamlit interface for the Smart Incident (vehicle-health) model
------------------------------------------------------------------------------------


Run locally:
    pip install streamlit scikit-learn pandas
    streamlit run dashboard.py
Opens at http://localhost:8501
"""
import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    with open("incident_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]

model, FEATURES = load_model()

# recommended action for each severity label
ACTIONS = {
    "Normal":   "Vehicle operating normally. No action needed.",
    "Warning":  "Schedule a service check soon.",
    "Critical": "Stop driving and inspect the vehicle immediately.",
}
COLORS = {"Normal": "green", "Warning": "orange", "Critical": "red"}

st.title("Smart Incident — Vehicle Health Monitor")
st.write("Adjust the sensor readings to see the predicted incident severity.")

col1, col2 = st.columns(2)
with col1:
    engine_temp = st.slider("Engine temperature (°C)", 70.0, 130.0, 95.0)
    oil_pressure = st.slider("Oil pressure (psi)", 10.0, 65.0, 40.0)
    brake_pad = st.slider("Brake pad remaining (mm)", 2.0, 12.0, 8.0)
with col2:
    battery = st.slider("Battery voltage (V)", 10.5, 14.5, 12.6)
    mileage = st.slider("Mileage (thousands)", 5.0, 200.0, 60.0)
    tire = st.slider("Tire pressure (psi)", 20.0, 45.0, 33.0)

if st.button("Check Vehicle Health"):
    row = pd.DataFrame([[engine_temp, oil_pressure, brake_pad, battery, mileage, tire]],
                       columns=FEATURES)

    # the model predicts the severity label directly as a string ("Normal", etc.)
    label = str(model.predict(row)[0])

    # confidence = the probability the model assigned to the predicted class
    proba = model.predict_proba(row)[0]
    confidence = proba[list(model.classes_).index(label)]

    color = COLORS.get(label, "gray")
    st.markdown(f"### Status: :{color}[{label}]")
    st.write(f"Model confidence: {confidence:.1%}")

    if label == "Critical":
        st.error(ACTIONS[label])
    elif label == "Warning":
        st.warning(ACTIONS[label])
    else:
        st.success(ACTIONS[label])