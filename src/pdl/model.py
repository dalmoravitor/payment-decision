"""Modelo calibrado. Fica em módulo próprio para o pickle ser carregável pela API
(classes definidas em __main__ não são encontradas ao desserializar)."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrated:
    """Modelo + calibração de Platt (regressão logística 1-D sobre o logit do score)."""

    def __init__(self, model):
        self.model = model
        self.platt = LogisticRegression(C=1e6, max_iter=1000)

    def fit_calibration(self, X_val, y_val):
        self.platt.fit(_logit(self.model.predict_proba(X_val)[:, 1]), y_val)
        return self

    def predict_proba_fraud(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        return self.platt.predict_proba(_logit(raw))[:, 1]
