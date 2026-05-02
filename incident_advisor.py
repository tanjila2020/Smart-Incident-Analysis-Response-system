def generate_incident_summary(severity, incident_data, operator_notes):
    """
    Generate a human-readable incident summary.

    This is a lightweight LLM-style reasoning layer.
    It converts model output and machine readings into plain-English guidance.
    """

    air_temp = incident_data["Air temperature [K]"]
    process_temp = incident_data["Process temperature [K]"]
    rotational_speed = incident_data["Rotational speed [rpm]"]
    torque = incident_data["Torque [Nm]"]
    tool_wear = incident_data["Tool wear [min]"]
    machine_type = incident_data["Type"]

    temp_gap = process_temp - air_temp

    risk_factors = []

    if temp_gap < 8.6:
        risk_factors.append(
            "low temperature gap between air and process temperature, which may indicate heat dissipation risk"
        )

    if rotational_speed < 1380:
        risk_factors.append(
            "low rotational speed, which may increase process instability"
        )

    if torque > 55:
        risk_factors.append(
            "high torque load, which may indicate mechanical strain"
        )

    if tool_wear > 200:
        risk_factors.append(
            "high tool wear, which may increase the likelihood of tool-related failure"
        )

    if not risk_factors:
        risk_factors.append(
            "no major individual risk factor was detected, but the model still identified a pattern worth reviewing"
        )

    if severity == "high":
        action = (
            "Stop or slow the machine if safe to do so, notify maintenance, "
            "inspect tooling, torque load, temperature conditions, and review recent process changes."
        )
    elif severity == "medium":
        action = (
            "Schedule maintenance review, monitor the machine closely, "
            "check tool wear, and verify whether operating conditions are trending toward failure."
        )
    else:
        action = (
            "Continue normal operation, but keep monitoring sensor trends and maintenance history."
        )

    summary = f"""
Incident Severity: {severity.upper()}

Machine Type: {machine_type}

Key Risk Factors:
- {"; ".join(risk_factors)}

Operator Notes:
{operator_notes if operator_notes else "No operator notes provided."}

Recommended Action:
{action}
"""

    return summary.strip()


def build_llm_prompt(severity, incident_data, operator_notes):
    """
    Optional helper prompt if you later connect this project to an actual LLM API.
    """

    prompt = f"""
You are an industrial maintenance assistant.

A machine learning model predicted the following incident severity:

Severity: {severity}

Machine readings:
- Machine type: {incident_data["Type"]}
- Air temperature: {incident_data["Air temperature [K]"]} K
- Process temperature: {incident_data["Process temperature [K]"]} K
- Rotational speed: {incident_data["Rotational speed [rpm]"]} rpm
- Torque: {incident_data["Torque [Nm]"]} Nm
- Tool wear: {incident_data["Tool wear [min]"]} min

Operator notes:
{operator_notes}

Explain the likely issue in plain English and recommend next steps.
Keep the response concise and practical.
"""

    return prompt.strip()