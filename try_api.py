import pandas as pd, requests
from src.pdl import data

df = pd.read_csv("data/creditcard.csv")
for classe in (0, 1):
    row = df[df.Class == classe].iloc[0]
    body = {f: float(row[f]) for f in data.FEATURES}
    print(classe, requests.post("http://127.0.0.1:8000/decision", json=body).json())