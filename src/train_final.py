import json
import joblib
import os
import pandas as pd
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def train_final_model():

    X_train = pd.read_parquet("data/features/X_train.parquet")
    X_test = pd.read_parquet("data/features/X_test.parquet")
    y_train = pd.read_parquet("data/features/y_train.parquet").values.ravel()
    y_test = pd.read_parquet("data/features/y_test.parquet").values.ravel()

    model = GradientBoostingClassifier(
        n_estimators=81,
        learning_rate=0.011566433635138113,
        max_depth=2
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.1).astype(int)  # seu melhor threshold

    f1 = f1_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    acc = accuracy_score(y_test, preds)

    print("\n=== FINAL MODEL ===")
    print(f"F1: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {acc}")

    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_experiment("final_model_gradient_boosting")
    with mlflow.start_run(run_name="final_model_gradient_boosting"):
        mlflow.log_param("model", "gradient_boosting")
        mlflow.log_params({
            "n_estimators": 81,
            "learning_rate": 0.011566433635138113,
            "max_depth": 2,
            "threshold": 0.1
        })
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/final_model.pkl")
        config = {
        "threshold": 0.1
        }

        with open("models/model_config.json", "w") as f:
            json.dump(config, f)