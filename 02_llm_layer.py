"""
02_llm_layer.py
---------------
The LLM layer: turns the classifier's severity prediction into a clear,
plain-language incident report (explanation + recommended action) using an LLM,
with PROMPT ENGINEERING.



Two modes:
    python3 02_llm_layer.py            # OFFLINE mock (always runs, no key needed)
    python3 02_llm_layer.py --real     # REAL LLM API (needs key + internet)

For the real mode:
    pip install anthropic
    export ANTHROPIC_API_KEY=[key here]
"""
import os, sys, json, pickle
import pandas as pd

HERE = os.path.dirname(__file__)
SEVERITY = {0: "Normal", 1: "Warning", 2: "Critical"}

# ----------------------------------------------------------------------
# Load the trained classifier
# ----------------------------------------------------------------------
def load_model():
    with open(os.path.join(HERE, "incident_model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]

# ----------------------------------------------------------------------
# PROMPT ENGINEERING -- the real deliverable of this layer
# ----------------------------------------------------------------------
def build_system_prompt():
    """Sets the role, task, tone, and hard rules (rule #1 fights hallucination)."""
    return (
        "You are a vehicle maintenance assistant. Your job is to explain a "
        "vehicle's health status to a driver or fleet operator who is NOT a "
        "technical expert.\n"
        "Rules you must follow:\n"
        "1. Base your explanation ONLY on the sensor readings and severity "
        "provided. Do not invent numbers or causes not supported by the data.\n"
        "2. Be concise and practical: plain language, no jargon.\n"
        "3. Always return your answer in the exact JSON format shown, no extra text.\n"
        "4. If severity is Critical, the recommended_action must convey urgency."
    )

def build_few_shot_examples():
    """Example input->output pairs teach the exact format and style."""
    return [
        {"role": "user", "content": (
            "Severity: Warning\n"
            "Sensor readings:\n"
            "- Engine temperature: 111 C\n- Oil pressure: 28 psi\n"
            "- Brake pad: 5 mm\n- Battery voltage: 12.3 V\n"
            "- Tire pressure: 33 psi\n"
            "Top contributing factors (from the model): engine temperature, oil pressure"
        )},
        {"role": "assistant", "content": json.dumps({
            "status": "Warning",
            "explanation": "The engine is running hotter than normal and oil pressure is on the low side, which together suggest the engine is under stress.",
            "recommended_action": "Schedule a service check soon and avoid heavy loads until then.",
            "urgency": "Medium"
        })},
        {"role": "user", "content": (
            "Severity: Normal\n"
            "Sensor readings:\n"
            "- Engine temperature: 93 C\n- Oil pressure: 42 psi\n"
            "- Brake pad: 9 mm\n- Battery voltage: 12.6 V\n"
            "- Tire pressure: 33 psi\n"
            "Top contributing factors (from the model): none significant"
        )},
        {"role": "assistant", "content": json.dumps({
            "status": "Normal",
            "explanation": "All readings are within their normal ranges. The vehicle is operating normally.",
            "recommended_action": "No action needed. Continue normal operation.",
            "urgency": "Low"
        })},
    ]

def build_user_message(row, severity_label, top_factors):
    """The real query: structured readings + the model's top factors, to ground it."""
    return (
        f"Severity: {severity_label}\n"
        "Sensor readings:\n"
        f"- Engine temperature: {row['engine_temp_C']:.0f} C\n"
        f"- Oil pressure: {row['oil_pressure_psi']:.0f} psi\n"
        f"- Brake pad: {row['brake_pad_mm']:.0f} mm\n"
        f"- Battery voltage: {row['battery_voltage_V']:.1f} V\n"
        f"- Tire pressure: {row['tire_pressure_psi']:.0f} psi\n"
        f"Top contributing factors (from the model): {top_factors}"
    )

def assemble_messages(row, severity_label, top_factors):
    return (
        [{"role": "system", "content": build_system_prompt()}]
        + build_few_shot_examples()
        + [{"role": "user", "content": build_user_message(row, severity_label, top_factors)}]
    )

# ----------------------------------------------------------------------
# Real LLM call (Anthropic)
# ----------------------------------------------------------------------
def call_real_llm(messages):
    from anthropic import Anthropic
    client = Anthropic()
    system = messages[0]["content"]
    convo = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        temperature=0,               # deterministic -> consistent, factual (prompt-eng choice)
        system=system,
        messages=convo,
    )
    return resp.content[0].text

# ----------------------------------------------------------------------
# Offline mock -- stand-in so the pipeline runs without a key.

# ----------------------------------------------------------------------
def call_mock_llm(severity_label, top_factors):
    templates = {
        "Critical": {
            "explanation": f"Readings indicate a serious problem (mainly {top_factors}). This points to a fault that can worsen quickly.",
            "recommended_action": "Stop driving and have the vehicle inspected immediately.",
            "urgency": "High"},
        "Warning": {
            "explanation": f"Some readings are outside their normal range (mainly {top_factors}), suggesting rising stress on the vehicle.",
            "recommended_action": "Schedule a service check soon and monitor the vehicle.",
            "urgency": "Medium"},
        "Normal": {
            "explanation": "All readings are within normal ranges. The vehicle is operating normally.",
            "recommended_action": "No action needed. Continue normal operation.",
            "urgency": "Low"},
    }
    return json.dumps({"status": severity_label, **templates[severity_label]})

# ----------------------------------------------------------------------
# top contributing factors (grounds the prompt in what drove the prediction)
# ----------------------------------------------------------------------
def top_factors(model, features, k=2):
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    factors = {"engine_temp_C":"engine temperature","oil_pressure_psi":"oil pressure",
              "brake_pad_mm":"brake wear","battery_voltage_V":"battery voltage",
              "tire_pressure_psi":"tire pressure","mileage_k":"mileage"}
    return ", ".join(factors.get(f, f) for f in imp.head(k).index)

# ----------------------------------------------------------------------
# End-to-end demo: classify -> build prompt -> LLM -> parsed report
# ----------------------------------------------------------------------
def run(use_real=False):
    model, features = load_model()
    df = pd.read_csv(os.path.join(HERE, "vehicle_health_clean.csv"))

    sample = pd.concat([
        df[df.severity == "Critical"].head(1),
        df[df.severity == "Warning"].head(1),
        df[df.severity == "Normal"].head(1),
    ])

    print("=" * 72)
    print(f"INCIDENT REPORTS  (mode: {'REAL API' if use_real else 'OFFLINE MOCK'})")
    print("=" * 72)

    for _, row in sample.iterrows():
        X_row = row[features].to_frame().T
        # the model predicts the severity label directly (e.g. "Warning")
        label = str(model.predict(X_row)[0])
        factors = "none significant" if label == "Normal" else top_factors(model, features)
        messages = assemble_messages(row, label, factors)

        raw = call_real_llm(messages) if use_real else call_mock_llm(label, factors)
        try:
            report = json.loads(raw)
        except json.JSONDecodeError:
            report = {"status": label, "explanation": raw,
                      "recommended_action": "(parse error)", "urgency": "?"}

        print(f"\nPredicted severity: {label}")
        print(f"  Explanation:  {report['explanation']}")
        print(f"  Action:       {report['recommended_action']}")
        print(f"  Urgency:      {report['urgency']}")
        print("-" * 72)

    # show a full prompt so the prompt engineering is visible
    print("\n" + "=" * 72)
    print("EXAMPLE OF THE FULL PROMPT SENT TO THE LLM (last case):")
    print("=" * 72)
    for m in messages:
        print(f"\n[{m['role'].upper()}]")
        print(m["content"][:600])

if __name__ == "__main__":
    run(use_real="--real" in sys.argv)
