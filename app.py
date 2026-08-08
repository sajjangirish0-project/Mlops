from fastapi import FastAPI
import joblib

app = FastAPI(title="Salary Prediction API")
model = joblib.load("models/model.pkl")

@app.get("/")
def root():
    return {"message": "Salary Prediction API is running"}

@app.get("/predict")
def predict(age: int, experience: int, education: int):
    prediction = model.predict([[age, experience, education]])
    return {"salary_high": int(prediction[0])}
