# src/utils.py

import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """
    Carrega um arquivo YAML e retorna como dicionário.

    Parâmetros:
        path (str): caminho para o arquivo YAML

    Retorno:
        dict: conteúdo do YAML
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)