"""
Automated Test Suite for the 11 Mandatory Business & System Scenarios.

Author: Elpidio Junior E-ABC
License: MIT
"""

import pytest
from typing import List, Dict, Any

from src.data.schemas import to_cents, to_currency, to_reais
from src.data.synthetic import SyntheticDataGenerator, DEFAULT_CUSTOMER_PROFILES
from src.solvers.solver_factory import get_solver
from src.features.engineering import FeatureExtractor
from src.models.baselines import BaselineModelsManager
from src.models.neural_mlp import DeepLearningRankingModel
from src.pipeline.inference import suggest_invoice_payments, validate_combination_integrity


@pytest.fixture(scope="session")
def trained_models_fixture():
    """Generates synthetic history and trains baseline + PyTorch models once for the test session."""
    gen = SyntheticDataGenerator(seed=42)
    history = gen.generate_payment_history(payments_per_profile=25, invoices_pool_size=12)

    solver = get_solver("backtracking", max_combinations=50)
    extractor = FeatureExtractor()
    df_X, y, meta = extractor.create_training_dataset(history, solver.solve)

    split_idx = int(len(df_X) * 0.8)
    X_train, y_train = df_X.iloc[:split_idx], y[:split_idx]
    X_test, y_test = df_X.iloc[split_idx:], y[split_idx:]
    meta_test = meta[split_idx:]

    baselines = BaselineModelsManager()
    baselines.fit_and_evaluate(X_train, y_train, X_test, y_test, meta_test)

    dl_model = DeepLearningRankingModel(epochs=30, batch_size=32)
    dl_model.fit(X_train, y_train)

    return {
        "generator": gen,
        "history": history,
        "baselines": baselines,
        "dl_model": dl_model,
        "extractor": extractor
    }


def test_scenario_01_single_exact_match():
    """Test 1: Single exact combination exists."""
    customer_id = 41
    amount = 5000.00
    invoices = [
        {"id": 101, "customer_id": 41, "value": 2000.00, "status": "OPEN"},
        {"id": 102, "customer_id": 41, "value": 3000.00, "status": "OPEN"},
        {"id": 103, "customer_id": 41, "value": 4500.00, "status": "OPEN"},
    ]

    res = suggest_invoice_payments(customer_id, amount, invoices)

    assert res["status"] == "EXACT_MATCH"
    assert len(res["combinations"]) == 1
    comb = res["combinations"][0]
    assert set(comb["invoice_ids"]) == {101, 102}
    assert comb["total"] == 5000.00
    assert comb["remaining"] == 0.00


def test_scenario_02_multiple_exact_matches():
    """Test 2: Multiple exact combinations exist."""
    customer_id = 41
    amount = 10000.00
    invoices = [
        {"id": 101, "customer_id": 41, "value": 2000.00, "status": "OPEN"},
        {"id": 102, "customer_id": 41, "value": 3500.00, "status": "OPEN"},
        {"id": 103, "customer_id": 41, "value": 1500.00, "status": "OPEN"},
        {"id": 104, "customer_id": 41, "value": 5000.00, "status": "OPEN"},
        {"id": 105, "customer_id": 41, "value": 5000.00, "status": "OPEN"},
        {"id": 106, "customer_id": 41, "value": 3000.00, "status": "OPEN"},
    ]

    res = suggest_invoice_payments(customer_id, amount, invoices)

    assert res["status"] == "EXACT_MATCH"
    assert len(res["combinations"]) >= 2
    for c in res["combinations"]:
        assert c["total"] == 10000.00
        assert c["remaining"] == 0.00
        assert sum(c["invoice_values"]) == 10000.00


def test_scenario_03_no_exact_match():
    """Test 3: No exact combination exists."""
    customer_id = 41
    amount = 10000.00
    invoices = [
        {"id": 101, "customer_id": 41, "value": 2000.00, "status": "OPEN"},
        {"id": 102, "customer_id": 41, "value": 3000.00, "status": "OPEN"},
        {"id": 103, "customer_id": 41, "value": 4000.00, "status": "OPEN"},
    ]

    res = suggest_invoice_payments(customer_id, amount, invoices)

    assert res["status"] == "NO_EXACT_MATCH"
    assert len(res["combinations"]) == 0


def test_scenario_04_other_customer_invoices_ignored():
    """Test 4: Invoices from other customers must be ignored."""
    customer_id = 41
    amount = 5000.00
    invoices = [
        {"id": 101, "customer_id": 41, "value": 2000.00, "status": "OPEN"},
        {"id": 102, "customer_id": 99, "value": 3000.00, "status": "OPEN"},  # Foreign customer
        {"id": 103, "customer_id": 41, "value": 3000.00, "status": "OPEN"},
    ]

    res = suggest_invoice_payments(customer_id, amount, invoices)

    assert res["status"] == "EXACT_MATCH"
    assert len(res["combinations"]) == 1
    comb = res["combinations"][0]
    assert set(comb["invoice_ids"]) == {101, 103}
    assert 102 not in comb["invoice_ids"]


def test_scenario_05_invoices_greater_than_amount():
    """Test 5: Invoices with values greater than amount are pruned."""
    customer_id = 41
    amount = 3000.00
    invoices = [
        {"id": 101, "customer_id": 41, "value": 10000.00, "status": "OPEN"},  # Greater than amount
        {"id": 102, "customer_id": 41, "value": 1500.00, "status": "OPEN"},
        {"id": 103, "customer_id": 41, "value": 1500.00, "status": "OPEN"},
    ]

    res = suggest_invoice_payments(customer_id, amount, invoices)

    assert res["status"] == "EXACT_MATCH"
    assert len(res["combinations"]) == 1
    comb = res["combinations"][0]
    assert set(comb["invoice_ids"]) == {102, 103}
    assert 101 not in comb["invoice_ids"]


def test_scenario_06_invoices_with_identical_values():
    """Test 6: Invoices with identical face values are distinguished by ID."""
    customer_id = 41
    amount = 4000.00
    invoices = [
        {"id": 101, "customer_id": 41, "value": 2000.00, "status": "OPEN"},
        {"id": 102, "customer_id": 41, "value": 2000.00, "status": "OPEN"},
        {"id": 103, "customer_id": 41, "value": 2000.00, "status": "OPEN"},
    ]

    res = suggest_invoice_payments(customer_id, amount, invoices)

    assert res["status"] == "EXACT_MATCH"
    assert len(res["combinations"]) == 3
    for c in res["combinations"]:
        assert len(c["invoice_ids"]) == 2
        assert c["total"] == 4000.00


def test_scenario_07_amount_equals_single_invoice():
    """Test 7: Amount equals face value of a single invoice."""
    customer_id = 41
    amount = 7500.00
    invoices = [
        {"id": 101, "customer_id": 41, "value": 2000.00, "status": "OPEN"},
        {"id": 102, "customer_id": 41, "value": 7500.00, "status": "OPEN"},
        {"id": 103, "customer_id": 41, "value": 3000.00, "status": "OPEN"},
    ]

    res = suggest_invoice_payments(customer_id, amount, invoices)

    assert res["status"] == "EXACT_MATCH"
    assert len(res["combinations"]) == 1
    comb = res["combinations"][0]
    assert comb["invoice_ids"] == [102]
    assert comb["number_of_invoices"] == 1
    assert comb["total"] == 7500.00


def test_scenario_08_amount_greater_than_total_debt():
    """Test 8: Amount is greater than total customer debt."""
    customer_id = 41
    amount = 50000.00
    invoices = [
        {"id": 101, "customer_id": 41, "value": 2000.00, "status": "OPEN"},
        {"id": 102, "customer_id": 41, "value": 3000.00, "status": "OPEN"},
    ]

    res = suggest_invoice_payments(customer_id, amount, invoices)

    assert res["status"] == "NO_EXACT_MATCH"
    assert len(res["combinations"]) == 0


def test_scenario_09_hundreds_of_invoices_performance():
    """Test 9: Hundreds of candidate open invoices."""
    customer_id = 41
    gen = SyntheticDataGenerator(seed=123)
    invoices = gen.generate_invoices_for_customer(customer_id=41, n_invoices=200, min_value=100.0, max_value=2000.0)

    target_cents = to_cents(invoices[0]["value"]) + to_cents(invoices[1]["value"]) + to_cents(invoices[2]["value"])
    amount = to_currency(target_cents)

    res = suggest_invoice_payments(
        customer_id=customer_id,
        amount=amount,
        invoices=invoices,
        solver_name="branch_and_bound",
        max_combinations=20
    )

    assert res["status"] == "EXACT_MATCH"
    assert len(res["combinations"]) >= 1
    assert res["combinations"][0]["total"] == amount


def test_scenario_10_customer_with_sufficient_history(trained_models_fixture):
    """Test 10: Customer has sufficient history for ML ranking."""
    ctx = trained_models_fixture
    customer_id = 41
    amount = 3000.00

    invoices = [
        {"id": 201, "customer_id": 41, "value": 1500.00, "due_date": "2026-01-10", "issue_date": "2025-12-01", "status": "OPEN"},
        {"id": 202, "customer_id": 41, "value": 1500.00, "due_date": "2026-01-15", "issue_date": "2025-12-05", "status": "OPEN"},
        {"id": 203, "customer_id": 41, "value": 1500.00, "due_date": "2026-04-10", "issue_date": "2026-02-01", "status": "OPEN"},
        {"id": 204, "customer_id": 41, "value": 1500.00, "due_date": "2026-04-15", "issue_date": "2026-02-05", "status": "OPEN"},
    ]

    res = suggest_invoice_payments(
        customer_id=customer_id,
        amount=amount,
        invoices=invoices,
        model=ctx["dl_model"],
        payment_history=ctx["history"],
        min_history_events=3
    )

    assert res["status"] == "EXACT_MATCH"
    assert len(res["combinations"]) >= 1
    top_comb = res["combinations"][0]
    assert 201 in top_comb["invoice_ids"] or 202 in top_comb["invoice_ids"]
    assert top_comb["score"] >= 0.5


def test_scenario_11_customer_cold_start_fallback():
    """Test 11: Customer without historical payments triggers deterministic fallback."""
    customer_id = 999
    amount = 4000.00

    invoices = [
        {"id": 301, "customer_id": 999, "value": 2000.00, "due_date": "2026-01-01", "issue_date": "2025-11-01", "status": "OPEN"},
        {"id": 302, "customer_id": 999, "value": 2000.00, "due_date": "2026-01-05", "issue_date": "2025-11-05", "status": "OPEN"},
        {"id": 303, "customer_id": 999, "value": 4000.00, "due_date": "2026-04-01", "issue_date": "2026-02-01", "status": "OPEN"},
    ]

    res = suggest_invoice_payments(
        customer_id=customer_id,
        amount=amount,
        invoices=invoices,
        model=None,
        payment_history=[],
        min_history_events=3
    )

    assert res["status"] == "EXACT_MATCH"
    assert len(res["combinations"]) == 2
    for c in res["combinations"]:
        assert any("fallback" in r.lower() or "rules" in r.lower() or "standard" in r.lower() for r in c.get("reasons", []))


def test_scenario_12_multi_currency_brl_converted_to_usd_with_receipt_date_and_quotation():
    """Test 12: Invoices in BRL converted to USD with receipt date and dollar exchange rate quotation."""
    customer_id = 41
    # Received amount: $3,000.00 USD on 2026-03-15 with USD quotation 5.00 BRL/USD
    amount = 3000.00
    receipt_date = "2026-03-15"
    exchange_rate = 5.00

    invoices = [
        # R$ 5,000.00 BRL / 5.0 = $1,000.00 USD
        {"id": 401, "customer_id": 41, "value": 5000.00, "currency": "BRL", "status": "OPEN"},
        # R$ 10,000.00 BRL / 5.0 = $2,000.00 USD
        {"id": 402, "customer_id": 41, "value": 10000.00, "currency": "BRL", "status": "OPEN"},
        # R$ 15,000.00 BRL / 5.0 = $3,000.00 USD
        {"id": 403, "customer_id": 41, "value": 15000.00, "currency": "BRL", "status": "OPEN"},
        # R$ 25,000.00 BRL / 5.0 = $5,000.00 USD (exceeds target $3,000 USD)
        {"id": 404, "customer_id": 41, "value": 25000.00, "currency": "BRL", "status": "OPEN"},
    ]

    res = suggest_invoice_payments(
        customer_id=customer_id,
        amount=amount,
        invoices=invoices,
        currency="USD",
        receipt_date=receipt_date,
        exchange_rate=exchange_rate
    )

    assert res["status"] == "EXACT_MATCH"
    assert res["currency"] == "USD"
    assert res["amount"] == 3000.00
    assert res["receipt_date"] == "2026-03-15"
    assert res["exchange_rate"] == 5.00
    assert len(res["combinations"]) == 2  # [401, 402] (1000 + 2000 = 3000) and [403] (3000)

    for combo in res["combinations"]:
        assert combo["currency"] == "USD"
        assert combo["total"] == 3000.00
        assert combo["remaining"] == 0.00
        assert "invoices_details" in combo
        for inv_detail in combo["invoices_details"]:
            assert inv_detail["currency"] == "USD"
            assert inv_detail["original_currency"] == "BRL"
            assert inv_detail["exchange_rate"] == 5.00
            assert inv_detail["value"] == round(inv_detail["original_value"] / 5.00, 2)


def test_scenario_13_mixed_currencies_matching_usd_target():
    """Test 13: Invoices in multiple currencies (USD, BRL, EUR) matching a USD target amount."""
    customer_id = 41
    amount = 5000.00
    receipt_date = "2026-03-20"
    exchange_rates = {
        "USD": 1.0,
        "BRL": 5.0,     # R$ 10,000 BRL -> $2,000 USD
        "EUR": 0.8      # 1,600 EUR / 0.8 -> $2,000 USD
    }

    invoices = [
        {"id": 501, "customer_id": 41, "value": 1000.00, "currency": "USD", "status": "OPEN"},
        {"id": 502, "customer_id": 41, "value": 10000.00, "currency": "BRL", "status": "OPEN"},
        {"id": 503, "customer_id": 41, "value": 1600.00, "currency": "EUR", "status": "OPEN"},
    ]

    res = suggest_invoice_payments(
        customer_id=customer_id,
        amount=amount,
        invoices=invoices,
        exchange_rates=exchange_rates,
        receipt_date=receipt_date,
        exchange_rate=5.0
    )

    assert res["status"] == "EXACT_MATCH"
    assert res["currency"] == "USD"
    assert res["amount"] == 5000.00
    assert len(res["combinations"]) == 1
    comb = res["combinations"][0]
    assert set(comb["invoice_ids"]) == {501, 502, 503}
    assert comb["total"] == 5000.00
    assert comb["currency"] == "USD"

