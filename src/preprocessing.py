# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
from src.utils import load_config


def run_preprocessing():
    pipeline_config = load_config("configs/pipeline.yaml")
    prep_config = load_config("configs/preprocessing.yaml")

    path = f"{pipeline_config['paths']['processed_data_dir']}/{pipeline_config['paths']['output_filename']}"

    df = pd.read_parquet(path)

    # Remover coluna inútil
    df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]

    # 🔴 SPLIT PRIMEIRO (evita leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 🔹 Imputação
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    # 🔹 Scaling
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # 🔹 Balanceamento (config-driven)
    balance_config = prep_config.get("dataset_balance", {})
    strategy = balance_config.get("strategy", "none")

    if strategy == "smote":
        print("Aplicando SMOTE...")

        from imblearn.over_sampling import SMOTE

        smote = SMOTE(
            sampling_strategy=balance_config.get("sampling_strategy", "auto"),
            k_neighbors=balance_config.get("k_neighbors", 5),
            random_state=balance_config.get("random_state", 42),
        )

        X_train, y_train = smote.fit_resample(X_train, y_train)

    elif strategy in [None, "none"]:
        print("SMOTE DESABILITADO")

    else:
        raise ValueError(f"Estratégia de balanceamento não suportada: {strategy}")

    print("Shape após SMOTE:", X_train.shape)
    print("Distribuição após SMOTE:", y_train.value_counts(normalize=True))

    output_dir = pipeline_config["paths"]["features_data_dir"]
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_parquet(f"{output_dir}/X_train.parquet")
    X_test.to_parquet(f"{output_dir}/X_test.parquet")
    y_train.to_frame().to_parquet(f"{output_dir}/y_train.parquet")
    y_test.to_frame().to_parquet(f"{output_dir}/y_test.parquet")

print("Dados salvos para modelagem")
if __name__ == "__main__":
    run_preprocessing()