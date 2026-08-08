import joblib

def test_model_prediction():
    model = joblib.load("models/model.pkl")
    prediction = model.predict([[35, 10, 2]])
    assert prediction[0] in [0, 1]
