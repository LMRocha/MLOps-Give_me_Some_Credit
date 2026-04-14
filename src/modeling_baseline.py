# src/modeling_baseline.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def run_baseline_model():
    # Carregar dados
    X_train = pd.read_parquet("data/features/X_train.parquet")
    X_test = pd.read_parquet("data/features/X_test.parquet")
    y_train = pd.read_parquet("data/features/y_train.parquet").values.ravel()
    y_test = pd.read_parquet("data/features/y_test.parquet").values.ravel()

    # MLflow setup
    mlflow.set_experiment("baseline-logistic-regression")

    with mlflow.start_run():

        # Modelo
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        # threshold = 0.4  # você vai testar vários valores
        # preds = (probs >= threshold).astype(int)

        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

        results = []

        for t in thresholds:
            preds = (probs >= t).astype(int)

            f1 = f1_score(y_test, preds)
            precision = precision_score(y_test, preds)
            recall = recall_score(y_test, preds)

            results.append((t, f1, precision, recall))

        # 🔹 Print organizado
        for t, f1, p, r in results:
            print(f"\nThreshold: {t}")
            print(f"F1: {f1}")
            print(f"Precision: {p}")
            print(f"Recall: {r}")
        
        # Métricas
        # f1 = f1_score(y_test, preds)
        # acc = accuracy_score(y_test, preds)
        # precision = precision_score(y_test, preds)
        # recall = recall_score(y_test, preds)
        best = max(results, key=lambda x: x[1])  # baseado em F1
        best_threshold, f1, precision, recall = best

        # Log
        mlflow.log_param("model", "logistic_regression")

        mlflow.log_metric("f1", f1)
        # mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_param("threshold", best_threshold)
        mlflow.sklearn.log_model(model, "model")

        print("\n=== BASELINE RESULTS ===")
        print(f"F1: {f1}")
        print(f"Accuracy: {accuracy_score}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")