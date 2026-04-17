# src/tuning.py

import optuna
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
import pandas as pd


def run_optuna():

    # carregar dados
    X_train = pd.read_parquet("data/features/X_train.parquet")
    X_test = pd.read_parquet("data/features/X_test.parquet")
    y_train = pd.read_parquet("data/features/y_train.parquet").values.ravel()
    y_test = pd.read_parquet("data/features/y_test.parquet").values.ravel()

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
        }

        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.1).astype(int)

        return f1_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best params:", study.best_params)
    print("Best F1:", study.best_value)

    # log mlflow
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_experiment("optuna_gradient_boosting")
    with mlflow.start_run(run_name="optuna_gradient_boosting"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("f1", study.best_value)
        
    return study