"""
01_train_model.py
------------------
Output: incident_model.pkl, vehicle_health_clean.csv
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "vehicle_health_raw.csv")

FEATURES = ["engine_temp_C", "oil_pressure_psi", "brake_pad_mm",
            "battery_voltage_V", "mileage_k", "tire_pressure_psi"]
TARGET = "severity"

# ======================================================================
# LOAD
# ======================================================================
if not os.path.exists(RAW):
    raise SystemExit("vehicle_health_raw.csv not found.")

df = pd.read_csv(RAW)
print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

# ======================================================================
# PRE-PROCESSING  
# ======================================================================


#  Drop the ID column  not a predictive feature.
if "vehicle_id" in df.columns:
    df = df.drop(columns=["vehicle_id"])
    print("1. Dropped non-predictive 'vehicle_id' column.")

#  Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"2. Removed {before - len(df)} duplicate rows.")

#  Standardize the label column 
df[TARGET] = df[TARGET].astype(str).str.strip().str.capitalize()
valid_labels = {"Normal", "Warning", "Critical"}
df = df[df[TARGET].isin(valid_labels)]
print(f"3. Standardized labels -> {sorted(df[TARGET].unique().tolist())}")

#  Fix impossible sensor values (glitches) by treating them as missing.

impossible = (
    (df["engine_temp_C"] < 60) | (df["engine_temp_C"] > 140) |
    (df["oil_pressure_psi"] < 0) | (df["oil_pressure_psi"] > 100)
)
n_bad = int(impossible.sum())
df.loc[df["engine_temp_C"] < 60, "engine_temp_C"] = np.nan
df.loc[df["engine_temp_C"] > 140, "engine_temp_C"] = np.nan
df.loc[df["oil_pressure_psi"] < 0, "oil_pressure_psi"] = np.nan
df.loc[df["oil_pressure_psi"] > 100, "oil_pressure_psi"] = np.nan
print(f"4. Flagged {n_bad} impossible sensor values as missing.")

#  Impute missing values with the column MEDIAN.
missing_before = int(df[FEATURES].isna().sum().sum())
for col in FEATURES:
    df[col] = df[col].fillna(df[col].median())
print(f"5. Imputed {missing_before} missing values with column medians.")

# save the cleaned data
clean_path = os.path.join(HERE, "vehicle_health_clean.csv")
df.to_csv(clean_path, index=False)
print(f"\nClean dataset: {df.shape[0]} rows -> vehicle_health_clean.csv")
print("Class distribution:")
print(df[TARGET].value_counts().to_string())

# ======================================================================
# TRAIN
# ======================================================================
print("\n--- Training ---")
X = df[FEATURES]
y = df[TARGET]

# time-independent classification, so a stratified split preserves class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

clf = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print("\nClassification report (test set):")
print(classification_report(y_test, pred))

imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Feature importance:")
for f, v in imp.items():
    print(f"  {f:20s} {v:.3f}")

with open(os.path.join(HERE, "incident_model.pkl"), "wb") as f:
    pickle.dump({"model": clf, "features": FEATURES}, f)
print("\nSaved incident_model.pkl")
