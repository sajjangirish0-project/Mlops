import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("data/employees.csv")

# Features and target
X = df[["age", "experience", "education"]]
y = df["salary_high"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Model parameters
n_estimators = 100
random_state = 42

# Start MLflow experiment
mlflow.set_experiment("salary-prediction")

with mlflow.start_run():

    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {accuracy:.2f}")

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    # Log model to MLflow
    mlflow.sklearn.log_model(
        model,
        "model"
    )

    print("Model saved to models/model.pkl")