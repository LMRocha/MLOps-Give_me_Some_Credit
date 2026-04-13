# src/utils.py
import yaml

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
# src/ingest.py

import os
import pandas as pd
from pathlib import Path
from utils import load_config
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(dataset: str, output_dir: str):
    api = KaggleApi()
    api.authenticate()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    api.dataset_download_files(
        dataset,
        path=output_dir,
        unzip=True
    )


def ingest_data():
    data_config = load_config("configs/data.yaml")
    pipeline_config = load_config("configs/pipeline.yaml")

    raw_dir = pipeline_config["paths"]["raw_data_dir"]
    processed_dir = pipeline_config["paths"]["processed_data_dir"]

    os.makedirs(processed_dir, exist_ok=True)

    # Download
    download_dataset(
        data_config["kaggle"]["dataset"],
        raw_dir
    )

    # Encontrar CSV
    file_pattern = data_config["kaggle"]["file_pattern"]
    files = list(Path(raw_dir).glob(file_pattern))

    if not files:
        raise ValueError("Nenhum arquivo encontrado")

    df = pd.read_csv(files[1])
    print(df.head())
    # Salvar parquet
    output_path = os.path.join(
        processed_dir,
        pipeline_config["paths"]["output_filename"]
    )
    print(f"Dados salvos em: {output_path}")

if __name__ == "__main__":
    ingest_data()