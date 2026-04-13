# src/ingest.py

import os
import pandas as pd
from pathlib import Path
from src.utils import load_config
from src.quality import run_quality_checks
from src.preprocessing import run_preprocessing
from kaggle.api.kaggle_api_extended import KaggleApi
from src.modeling_baseline import run_baseline_model


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

    # Salvar parquet
    output_path = os.path.join(
        processed_dir,
        pipeline_config["paths"]["output_filename"]
    )

    df.to_parquet(output_path, index=False)

    print(f"Dados salvos em: {output_path}")


if __name__ == "__main__":
    # ingest_data()
    run_quality_checks()
    run_preprocessing()
    run_baseline_model()