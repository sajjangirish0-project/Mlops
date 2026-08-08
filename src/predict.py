import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

data = pd.DataFrame(
    [[35, 10, 2]],
    columns=["age", "experience", "education"]
)

prediction = model.predict(data)

print("Salary prediction:", "High" if prediction[0] == 1 else "Low")