"""Treina modelos, calibra probabilidades e compara políticas de decisão no teste.

Fluxo:
  treino  -> ajusta o modelo
  validação -> (a) calibra probabilidades (Platt) e (b) escolhe o limiar global
  teste   -> avaliação final, tocado uma única vez
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import costs, data
from .model import Calibrated

ROOT = Path(__file__).resolve().parents[2]
CFG = costs.CostConfig()
RNG = np.random.default_rng(42)


def build_models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "hist_gb": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=42
        ),
    }


def bootstrap_saving(y, amount, rc_policy, n=2000):
    """IC 95% da economia vs. 'aprovar tudo', reamostrando transações do teste."""
    rc_base = costs.row_costs(y, amount, np.ones(len(y), bool), CFG)
    diff = rc_base - rc_policy
    idx = RNG.integers(0, len(diff), size=(n, len(diff)))
    sav = diff[idx].sum(axis=1)
    return float(diff.sum()), [float(np.percentile(sav, 2.5)), float(np.percentile(sav, 97.5))]


def evaluate(name, model, tr, va, te):
    model.fit(tr[data.FEATURES], tr[data.TARGET])
    cal = Calibrated(model).fit_calibration(va[data.FEATURES], va[data.TARGET])
    p_va = cal.predict_proba_fraud(va[data.FEATURES])
    p_te = cal.predict_proba_fraud(te[data.FEATURES])
    y_te, a_te = te[data.TARGET].values, te["Amount"].values

    # Política A: limiar global na probabilidade, escolhido na validação
    sweep_va = costs.sweep_thresholds(va[data.TARGET], va["Amount"], p_va, CFG)
    t = costs.best_threshold(sweep_va)["threshold"]
    act_A = np.where(p_te < t, costs.APPROVE, costs.DECLINE)
    # Política B: custo esperado, 3 ações (aprovar/revisar/recusar), sensível ao valor
    act_B = costs.best_action(p_te, a_te, CFG)

    policies = {}
    yb = y_te.astype(bool)
    for label, act in [("A_limiar_global", act_A), ("B_custo_esperado_3_acoes", act_B)]:
        rc = costs.row_costs_3way(y_te, a_te, act, CFG)
        saving, ci = bootstrap_saving(y_te, a_te, rc)
        appr = act == costs.APPROVE
        policies[label] = {
            "cost": float(rc.sum()),
            "saving_vs_approve_all": saving,
            "saving_ci95": ci,
            "approve_rate": float(appr.mean()),
            "review_rate": float((act == costs.REVIEW).mean()),
            "decline_rate": float((act == costs.DECLINE).mean()),
            "fraud_rate_among_approved": costs.fraud_rate_among_approved(y_te, appr),
            "frauds_declined": int((yb & (act == costs.DECLINE)).sum()),
            "frauds_reviewed": int((yb & (act == costs.REVIEW)).sum()),
            "frauds_approved": int((yb & appr).sum()),
            "legit_declined": int((~yb & (act == costs.DECLINE)).sum()),
        }
    result = {
        "model": name,
        "val_pr_auc": float(average_precision_score(va[data.TARGET], p_va)),
        "threshold_A_from_val": None if np.isinf(t) else float(t),
        "test_roc_auc": float(roc_auc_score(y_te, p_te)),
        "test_pr_auc": float(average_precision_score(y_te, p_te)),
        "test_cost_approve_all": costs.decision_cost(y_te, a_te, np.ones(len(y_te), bool), CFG),
        "policies": policies,
        "_sweep": costs.sweep_thresholds(y_te, a_te, p_te, CFG),
    }
    return result, cal


def plot_tradeoff(results, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        xs = [p["approval_rate"] * 100 for p in r["_sweep"]]
        ys = [p["fraud_rate_approved"] * 100 for p in r["_sweep"]]
        ax.plot(xs, ys, label=r["model"])
    ax.set_xlabel("Taxa de aprovação (%)")
    ax.set_ylabel("Fraude entre aprovadas (%)")
    ax.set_title("Trade-off aprovação × fraude (conjunto de teste)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    df = data.load()
    tr, va, te = data.temporal_split(df)
    results, fitted = [], {}
    for name, model in build_models().items():
        r, cal = evaluate(name, model, tr, va, te)
        results.append(r)
        fitted[name] = cal

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    summary = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(summary, indent=2))
    plot_tradeoff(results, ROOT / "reports" / "tradeoff.png")

    # Modelo da API escolhido pelo PR-AUC na VALIDAÇÃO (o teste não participa da escolha).
    best = max(results, key=lambda r: r["val_pr_auc"])
    joblib.dump({"calibrated": fitted[best["model"]], "name": best["model"]}, ROOT / "models" / "model.joblib")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
