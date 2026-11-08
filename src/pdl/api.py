"""API de decisão: recebe uma transação, devolve APPROVE / REVIEW / DECLINE com o score."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import costs
from .data import FEATURES

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.joblib"
CFG = costs.CostConfig()

app = FastAPI(title="Payment Decision Lab")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Modelo não treinado. Rode: python -m src.pdl.train")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class Transaction(BaseModel):
    Amount: float = Field(ge=0)
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float


def decide(p_fraud: float, amount: float) -> str:
    """Escolhe a ação de menor custo esperado (aprovar / revisar / recusar)."""
    action = int(costs.best_action(np.array([p_fraud]), np.array([amount]), CFG)[0])
    return costs.ACTION_NAMES[action]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/decision")
def decision(tx: Transaction):
    b = get_bundle()
    row = pd.DataFrame([tx.model_dump()])[FEATURES]
    p = float(b["calibrated"].predict_proba_fraud(row)[0])
    return {"decision": decide(p, tx.Amount), "p_fraud": p, "model": b["name"]}
