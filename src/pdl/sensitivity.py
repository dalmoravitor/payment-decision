"""Sensibilidade da economia às hipóteses de custo (margem, taxa de chargeback, taxa de revisão).

Mostra quando a política de custo esperado deixa de compensar. Rode depois de editar costs.py.
"""
import itertools

import numpy as np

from . import costs, data, train
from .model import Calibrated


def main():
    df = data.load()
    tr, va, te = data.temporal_split(df)
    model = train.build_models()["hist_gb"]
    model.fit(tr[data.FEATURES], tr[data.TARGET])
    cal = Calibrated(model).fit_calibration(va[data.FEATURES], va[data.TARGET])
    p = cal.predict_proba_fraud(te[data.FEATURES])
    y, a = te[data.TARGET].values, te["Amount"].values
    print(f"{'margem':>7} {'chargeback':>10} {'revisão':>8} | {'economia vs aprovar tudo':>26}")
    for margin, fee, rev in itertools.product([0.1, 0.2, 0.4], [5.0, 15.0, 50.0], [2.0, 5.0, 20.0]):
        cfg = costs.CostConfig(chargeback_fee=fee, margin=margin, review_fee=rev)
        act = costs.best_action(p, a, cfg)
        base = costs.row_costs(y, a, np.ones(len(y), bool), cfg).sum()
        pol = costs.row_costs_3way(y, a, act, cfg).sum()
        print(f"{margin:>7} {fee:>10} {rev:>8} | {base - pol:>26.0f}")


if __name__ == "__main__":
    main()
