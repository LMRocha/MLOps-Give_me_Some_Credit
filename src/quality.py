# src/quality.py

import pandas as pd
from src.utils import load_config


def run_quality_checks():
    pipeline_config = load_config("configs/pipeline.yaml")

    processed_dir = pipeline_config["paths"]["processed_data_dir"]
    filename = pipeline_config["paths"]["output_filename"]

    file_path = f"{processed_dir}/{filename}"

    df = pd.read_parquet(file_path)
    print(df.head())  # Apenas para verificar que o arquivo foi carregado corretamente
    print("\n=== QUALIDADE DOS DADOS ===\n")

    # 1. Shape
    print(f"Shape: {df.shape}")

    # 2. Tipos
    print("\nTipos de dados:")
    print(df.dtypes)

    # 3. Nulos
    print("\nValores nulos:")
    print(df.isnull().sum())

    # 4. Estatísticas
    print("\nEstatísticas descritivas:")
    print(df.describe())

    # 5. Target balance
    print("\nDistribuição do target:")
    print(df["SeriousDlqin2yrs"].value_counts(normalize=True))

    print("\n=== FIM DA ANÁLISE ===\n")

if __name__ == "__main__":
    run_quality_checks()