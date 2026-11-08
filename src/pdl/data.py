"""Carregamento e split temporal do dataset (ULB credit card fraud)."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "creditcard.csv"
# "Time" fica fora das features: é segundos desde o início de 2 dias de dados,
# não generaliza. Usamos só para ordenar e fazer o split temporal.
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Baixe o dataset 'Credit Card Fraud Detection' "
            "(ULB) no Kaggle e coloque em data/creditcard.csv"
        )
    return pd.read_csv(path).sort_values("Time").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train=0.6, val=0.2):
    """Split por tempo: treino < validação < teste.

    Em fraude, split aleatório vaza informação do futuro. A validação serve
    para escolher o limiar; o teste só é tocado uma vez, no fim.
    """
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]
