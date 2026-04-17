from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd

app = FastAPI()

# 🔹 carregar modelo do MLflow
model = mlflow.sklearn.load_model("runs:/111e96c401c8430ca94deaea1b65e744/model")


@app.get("/")
def home():
    return {"message": "API de risco de crédito rodando"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prob = model.predict_proba(df)[:, 1][0]
    pred = int(prob >= 0.1)

    return {
        "probability": float(prob),
        "prediction": pred
    }