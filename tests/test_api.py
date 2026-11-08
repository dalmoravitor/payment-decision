from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.pdl import data
from src.pdl.api import MODEL_PATH, app

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="rode o treino antes")
client = TestClient(app)


def _row(is_fraud: bool):
    df = data.load()
    r = df[df[data.TARGET] == int(is_fraud)].iloc[0]
    return {f: float(r[f]) for f in data.FEATURES}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_decision_shape_and_legit_is_approved():
    out = client.post("/decision", json=_row(False)).json()
    assert set(out) == {"decision", "p_fraud", "model"}
    assert out["decision"] == "APPROVE"


def test_negative_amount_rejected():
    row = _row(False)
    row["Amount"] = -1
    assert client.post("/decision", json=row).status_code == 422
