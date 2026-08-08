# MLOps Salary Prediction

Beginner-friendly MLOps project.

Flow:
Dataset -> Training -> Evaluation -> Model -> API -> Docker -> CI/CD

## Run locally

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
python src/predict.py

## Run API

uvicorn app:app --reload

Then open:
http://127.0.0.1:8000

Prediction:
http://127.0.0.1:8000/predict?age=35&experience=10&education=2

## Docker

docker build -t salary-prediction .
docker run -p 8000:8000 salary-prediction
