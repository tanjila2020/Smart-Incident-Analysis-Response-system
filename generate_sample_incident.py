import pandas as pd


sample_incidents = pd.DataFrame([
    {
        "Type": "L",
        "Air temperature [K]": 298.4,
        "Process temperature [K]": 306.1,
        "Rotational speed [rpm]": 1320,
        "Torque [Nm]": 62.5,
        "Tool wear [min]": 215
    },
    {
        "Type": "M",
        "Air temperature [K]": 300.2,
        "Process temperature [K]": 309.7,
        "Rotational speed [rpm]": 1485,
        "Torque [Nm]": 42.1,
        "Tool wear [min]": 120
    },
    {
        "Type": "H",
        "Air temperature [K]": 297.8,
        "Process temperature [K]": 305.4,
        "Rotational speed [rpm]": 1250,
        "Torque [Nm]": 58.8,
        "Tool wear [min]": 230
    }
])

sample_incidents.to_csv("sample_incidents.csv", index=False)

print("Created sample_incidents.csv")