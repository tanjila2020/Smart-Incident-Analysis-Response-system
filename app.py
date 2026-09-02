"""
app.py  --  FastAPI service for the Smart Incident (vehicle-health) severity model
-----------------------------------------------------------------------------------
Serves the trained model as a REST API. Send sensor readings in, get a predicted
severity + recommended action back.

Run locally:
    pip install fastapi uvicorn scikit-learn pandas
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000/docs
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="Smart Incident Severity API", version="1.0")

with open("incident_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]
FEATURES = bundle["features"]

ACTIONS = {
    "Normal":   "Vehicle operating normally. No action needed.",
    "Warning":  "Schedule a service check soon.",
    "Critical": "Stop driving and inspect the vehicle immediately.",
}

class VehicleReading(BaseModel):
    engine_temp_C: float
    oil_pressure_psi: float
    brake_pad_mm: float
    battery_voltage_V: float
    mileage_k: float
    tire_pressure_psi: float

@app.get("/")
def health():
    return {"status": "ok", "message": "Smart Incident API is running"}

@app.post("/predict")
def predict(reading: VehicleReading):
    row = pd.DataFrame([[
        reading.engine_temp_C,
        reading.oil_pressure_psi,
        reading.brake_pad_mm,
        reading.battery_voltage_V,
        reading.mileage_k,
        reading.tire_pressure_psi,
    ]], columns=FEATURES)

    # the model predicts the severity label directly as a string
    label = str(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    confidence = float(proba[list(model.classes_).index(label)])

    return {
        "severity": label,
        "confidence": round(confidence, 3),
        "recommended_action": ACTIONS[label],
    }