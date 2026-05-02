import joblib
import pandas as pd

# from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def create_severity_label(row):
    """
    Create a practical incident severity label from the failure columns.

    This is not meant to be a perfect industrial safety rule.
    It is a reasonable project-level severity mapping for ML demonstration.
    """

    if row["Machine failure"] == 0:
        return "low"

    # More severe failure types
    if row["HDF"] == 1 or row["PWF"] == 1 or row["OSF"] == 1:
        return "high"

    # Tool wear failure or random failure
    if row["TWF"] == 1 or row["RNF"] == 1:
        return "medium"

    return "medium"


def load_dataset():
    """
    Load the AI4I 2020 Predictive Maintenance Dataset from UCI.
    """
    df = pd.read_csv("ai4i2020.csv")

    return df


def train_model():
    df = load_dataset()

    df["severity"] = df.apply(create_severity_label, axis=1)

    features = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X = df[features]
    y = df["severity"]

    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    categorical_features = ["Type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print("Model Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    joblib.dump(pipeline, "incident_severity_model.pkl")

    print("\nModel saved as incident_severity_model.pkl")


if __name__ == "__main__":
    train_model()