"""Função de custo da decisão de pagamento.

>>> DECISÃO PARA O VITOR <<<
Os números abaixo são hipóteses, não fatos. Defenda-os no README.
- Fraude aprovada: perde-se o valor da venda + taxa fixa de chargeback.
- Venda legítima recusada: perde-se a margem da venda (e algo de confiança do cliente).
Mude CostConfig, rode de novo e veja como o limiar ótimo se move.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostConfig:
    chargeback_fee: float = 15.0   # taxa fixa por chargeback (hipótese)
    margin: float = 0.20           # margem sobre venda legítima recusada (hipótese)
    review_fee: float = 5.0        # custo fixo de uma revisão manual; revisor acerta sempre (hipótese)


def row_costs(y_true, amount, approve, cfg: CostConfig = CostConfig()):
    """Custo de cada decisão individual (approve=True aprova a venda)."""
    y_true = np.asarray(y_true).astype(bool)
    amount = np.asarray(amount, dtype=float)
    approve = np.asarray(approve).astype(bool)
    fraud_approved = y_true & approve
    legit_declined = ~y_true & ~approve
    return (
        fraud_approved * (amount + cfg.chargeback_fee)
        + legit_declined * (amount * cfg.margin)
    )


def decision_cost(y_true, amount, approve, cfg: CostConfig = CostConfig()) -> float:
    """Custo total de um conjunto de decisões."""
    return float(row_costs(y_true, amount, approve, cfg).sum())


def expected_costs(p_fraud, amount, cfg: CostConfig = CostConfig()):
    """Custo esperado de aprovar vs. recusar, dada a probabilidade de fraude p.

    aprovar: com prob. p é fraude -> perde (valor + taxa de chargeback)
    recusar: com prob. (1-p) era legítima -> perde a margem
    """
    p = np.asarray(p_fraud, dtype=float)
    amount = np.asarray(amount, dtype=float)
    cost_approve = p * (amount + cfg.chargeback_fee)
    cost_decline = (1 - p) * amount * cfg.margin
    return cost_approve, cost_decline


APPROVE, REVIEW, DECLINE = 0, 1, 2
ACTION_NAMES = {APPROVE: "APPROVE", REVIEW: "REVIEW", DECLINE: "DECLINE"}


def row_costs_3way(y_true, amount, action, cfg: CostConfig = CostConfig()):
    """Custo por transação com 3 ações. Revisão manual resolve a incerteza: custa só a taxa."""
    y_true = np.asarray(y_true).astype(bool)
    amount = np.asarray(amount, dtype=float)
    action = np.asarray(action)
    return (
        ((action == APPROVE) & y_true) * (amount + cfg.chargeback_fee)
        + ((action == DECLINE) & ~y_true) * (amount * cfg.margin)
        + (action == REVIEW) * cfg.review_fee
    )


def best_action(p_fraud, amount, cfg: CostConfig = CostConfig()):
    """Escolhe a ação de menor custo esperado entre aprovar, revisar e recusar.

    Só é ótima se p_fraud estiver CALIBRADA (ver train.py).
    """
    ca, cd = expected_costs(p_fraud, amount, cfg)
    stacked = np.stack([ca, np.full_like(ca, cfg.review_fee), cd])
    return stacked.argmin(axis=0)


def bayes_approve(p_fraud, amount, cfg: CostConfig = CostConfig()):
    """Regra de decisão sensível ao valor: aprova se aprovar custa menos que recusar.

    Só é ótima se p_fraud estiver CALIBRADA (ver train.py).
    """
    ca, cd = expected_costs(p_fraud, amount, cfg)
    return ca <= cd


def approval_rate(approve) -> float:
    return float(np.mean(np.asarray(approve).astype(bool)))


def fraud_rate_among_approved(y_true, approve) -> float:
    y_true = np.asarray(y_true).astype(bool)
    approve = np.asarray(approve).astype(bool)
    if approve.sum() == 0:
        return 0.0
    return float(y_true[approve].mean())


def sweep_thresholds(y_true, amount, score, cfg: CostConfig = CostConfig(), n=300):
    """Para cada limiar t, aprova se score < t. Retorna lista de dicts.

    A grade é fina na cauda: testa recusar de 0% até ~20% das transações em
    escala geométrica, porque a fraude é ~0,1% e uma grade grossa não a enxerga.
    """
    score = np.asarray(score)
    decline_fracs = np.concatenate([[0.0], np.geomspace(1e-4, 0.2, n)])
    out = []
    for f in decline_fracs:
        t = np.inf if f == 0 else float(np.quantile(score, 1 - f))
        approve = score < t
        out.append(
            {
                "threshold": t,
                "cost": decision_cost(y_true, amount, approve, cfg),
                "approval_rate": approval_rate(approve),
                "fraud_rate_approved": fraud_rate_among_approved(y_true, approve),
            }
        )
    return out


def best_threshold(sweep):
    return min(sweep, key=lambda r: r["cost"])
