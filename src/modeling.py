# src/modeling_baseline.py

import os

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from src.utils import load_config
import importlib


def run_modeling():
    config = load_config("configs/modeling.yaml")
    pipeline_config = load_config("configs/pipeline.yaml")

    data_dir = pipeline_config["paths"]["features_data_dir"]

    X_train = pd.read_parquet(f"{data_dir}/X_train.parquet")
    X_test = pd.read_parquet(f"{data_dir}/X_test.parquet")
    y_train = pd.read_parquet(f"{data_dir}/y_train.parquet").values.ravel()
    y_test = pd.read_parquet(f"{data_dir}/y_test.parquet").values.ravel()
    # MLflow setup
    
    models_config = config["models"]
    all_results = []
    for model_name, model_cfg in models_config.items():

        if not model_cfg.get("enabled", False):
            continue

        print(f"\n=== Treinando {model_name} ===")

        module = importlib.import_module(model_cfg["module"])
        ModelClass = getattr(module, model_cfg["class"])

        params = model_cfg.get("default_params", {})
        model = ModelClass(**params)

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        threshold = 0.1
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        acc = accuracy_score(y_test, preds)

        all_results.append({
        "model": model_name,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": acc
        })

        mlflow.set_experiment(config["modeling"]["experiment_name"])

        with mlflow.start_run(run_name=model_name):

            mlflow.log_param("model", model_name)
            mlflow.log_param("threshold", threshold)

            for k, v in params.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("f1", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="f1", ascending=False)
  
    print("\n=== COMPARAÇÃO FINAL ===")
    print("\n=== COMPARAÇÃO FINAL ===")
    print(results_df)
  
    best_model_name = "gradient_boosting"
  
    mlflow.log_param("best_model", best_model_name)
  
    os.makedirs("outputs", exist_ok=True)
  
    results_df.to_csv("outputs/model_comparison.csv", index=False)
  
    mlflow.log_artifact("outputs/model_comparison.csv")