import numpy as np

from src.pdl import costs
from src.pdl.api import decide

CFG = costs.CostConfig(chargeback_fee=10.0, margin=0.5, review_fee=5.0)


def test_approving_fraud_costs_amount_plus_fee():
    assert costs.decision_cost([1], [100.0], [True], CFG) == 110.0


def test_declining_legit_costs_margin():
    assert costs.decision_cost([0], [100.0], [False], CFG) == 50.0


def test_correct_decisions_cost_zero():
    assert costs.decision_cost([1, 0], [100.0, 40.0], [False, True], CFG) == 0.0


def test_review_costs_fixed_fee_regardless_of_truth():
    rc = costs.row_costs_3way([1, 0], [900.0, 900.0], [costs.REVIEW, costs.REVIEW], CFG)
    assert list(rc) == [5.0, 5.0]


def test_best_action_extremes():
    # p ~ 0 -> aprovar; p ~ 1 em venda pequena -> recusar
    assert costs.best_action([0.0001], [50.0], CFG)[0] == costs.APPROVE
    assert costs.best_action([0.99], [50.0], CFG)[0] == costs.DECLINE


def test_large_amount_with_uncertainty_goes_to_review():
    # venda enorme, p moderado: recusar custaria margem alta, aprovar risco alto -> revisar
    assert costs.best_action([0.2], [20000.0], CFG)[0] == costs.REVIEW


def test_decide_returns_names():
    assert decide(0.0001, 50.0) == "APPROVE"
    assert decide(0.99, 50.0) == "DECLINE"
    assert decide(0.2, 20000.0) == "REVIEW"


def test_best_threshold_is_min_cost():
    sw = [{"cost": 3}, {"cost": 1}, {"cost": 2}]
    assert costs.best_threshold(sw)["cost"] == 1


def test_approval_rate():
    assert np.isclose(costs.approval_rate([True, False, True, True]), 0.75)
