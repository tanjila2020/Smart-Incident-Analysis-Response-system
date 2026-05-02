import joblib
import pandas as pd
import streamlit as st

from incident_advisor import generate_incident_summary, build_llm_prompt


st.set_page_config(
    page_title="Smart Incident Analysis System",
    page_icon="🏭",
    layout="centered"
)

st.title("🏭 Smart Incident Analysis & Response Recommendation System")

st.write(
    "This app predicts machine incident severity using a machine learning model "
    "and generates a plain-English response recommendation using an LLM-style reasoning layer."
)

st.sidebar.header("Machine Readings")

machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])

air_temperature = st.sidebar.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=298.0
)

process_temperature = st.sidebar.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=308.0
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed [rpm]",
    min_value=500,
    max_value=3000,
    value=1350
)

torque = st.sidebar.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=55.0
)

tool_wear = st.sidebar.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=200
)

operator_notes = st.text_area(
    "Operator Notes",
    placeholder="Example: Machine vibration increased and operator noticed unusual heat near motor."
)

incident_data = {
    "Type": machine_type,
    "Air temperature [K]": air_temperature,
    "Process temperature [K]": process_temperature,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}

if st.button("Analyze Incident"):
    try:
        model = joblib.load("incident_severity_model.pkl")

        input_df = pd.DataFrame([incident_data])

        severity = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_

        probability_table = pd.DataFrame({
            "Severity": classes,
            "Probability": probabilities
        }).sort_values(by="Probability", ascending=False)

        st.subheader("Predicted Severity")

        if severity == "high":
            st.error(f"HIGH")
        elif severity == "medium":
            st.warning(f"MEDIUM")
        else:
            st.success(f"LOW")

        st.subheader("Prediction Confidence")
        st.dataframe(probability_table)

        st.subheader("Incident Analysis & Recommended Response")
        summary = generate_incident_summary(severity, incident_data, operator_notes)
        st.text_area("Generated Response", value=summary, height=300)

        st.subheader("Optional LLM Prompt")
        st.write(
            "If you later connect this project to an actual LLM API, this prompt can be sent to the model."
        )
        prompt = build_llm_prompt(severity, incident_data, operator_notes)
        st.text_area("LLM Prompt", value=prompt, height=300)

    except FileNotFoundError:
        st.error(
            "Model file not found. Run `python train_model.py` first to train and save the model."
        )