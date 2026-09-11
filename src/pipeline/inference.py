"""
Main Inference and Independent Accounting Integrity Verification Pipeline: suggest_invoice_payments.

Author: Elpidio Junior E-ABC
License: MIT
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from src.data.schemas import (
    to_cents,
    to_currency,
    to_reais,
    CombinationResult,
    SuggestionResponse
)
from src.solvers.solver_factory import get_solver
from src.features.engineering import FeatureExtractor
from src.models.fallback import DeterministicFallbackRanker
from src.explainability.explainer import SuggestionExplainer


def validate_combination_integrity(
    combination: List[Dict[str, Any]],
    expected_customer_id: Optional[int],
    expected_amount_cents: int
) -> bool:
    """
    Independent accounting verification prior to presenting combinations.
    Guarantees:
    1. Correct customer_id on all invoices (if customer_id specified);
    2. Zero duplicate invoices;
    3. All invoices in 'OPEN' status;
    4. Exact integer cents sum == expected_amount_cents;
    5. Zero remaining balance.
    """
    if not combination:
        return False

    seen_ids = set()
    total_cents = 0

    for inv in combination:
        inv_id = inv.get("id")
        if inv_id in seen_ids:
            return False
        seen_ids.add(inv_id)

        if expected_customer_id is not None and inv.get("customer_id") != expected_customer_id:
            return False

        if inv.get("status", "OPEN").upper() != "OPEN":
            return False

        val_cents = to_cents(inv.get("value", 0.0))
        if val_cents <= 0:
            return False
        total_cents += val_cents

    return total_cents == expected_amount_cents


def find_deterministic_matches(
    amount: float,
    invoices: List[Dict[str, Any]],
    customer_id: Optional[int] = None,
    currency: str = "USD",
    exchange_rate: Optional[float] = None,
    exchange_rates: Optional[Dict[str, float]] = None,
    receipt_date: Optional[str] = None,
    solver_name: str = "auto",
    max_combinations: int = 50,
    ranking_strategy: str = "fewest_invoices",
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Pure deterministic mathematical matching entry point for external platforms & microservices.
    Requires NO machine learning, neural networks, or historical data.

    Args:
        amount: Total received monetary amount (in USD or foreign currency converted to USD).
        invoices: List of candidate invoices (can be in USD, BRL, EUR, etc.).
        customer_id: Optional customer identifier to filter invoices. If None, considers all open invoices.
        currency: Currency of the received payment (default 'USD').
        exchange_rate: Dollar exchange rate quotation on the receipt date.
        exchange_rates: Optional dictionary mapping currency codes to exchange rates.
        receipt_date: Settlement / payment receipt date (YYYY-MM-DD).
        solver_name: Algorithm ('auto', 'branch_and_bound', 'backtracking', 'dynamic_programming').
        max_combinations: Maximum number of exact combinations to find (default 50).
        ranking_strategy: Deterministic ordering strategy ('fewest_invoices', 'overdue_first', 'highest_value_first', 'oldest_first', 'none').

    Returns:
        Structured dictionary with status, amount (USD), currency ('USD'), receipt_date,
        exchange_rate, combinations_count, and ranked combinations with full invoice details.
    """
    receipt_date = receipt_date or kwargs.get("payment_date") or kwargs.get("data_recebimento") or datetime.now().strftime("%Y-%m-%d")
    exchange_rate = exchange_rate or kwargs.get("cotacao") or kwargs.get("fx_rate")

    curr_upper = (currency or "USD").upper()
    if curr_upper != "USD":
        fx = exchange_rate or (exchange_rates.get(curr_upper) if exchange_rates else None) or 1.0
        if fx <= 0:
            fx = 1.0
        amount_usd = round(amount / fx, 2)
    else:
        amount_usd = round(amount, 2)

    if amount_usd <= 0 or not invoices:
        return {
            "status": "NO_EXACT_MATCH",
            "customer_id": customer_id,
            "amount": amount_usd,
            "currency": "USD",
            "receipt_date": receipt_date,
            "exchange_rate": exchange_rate,
            "combinations_count": 0,
            "combinations": [],
            "solver_used": "None"
        }

    amount_cents = to_cents(amount_usd)

    # 1. Filter, sanitize, and convert invoices to USD
    valid_invoices = []
    seen_ids = set()
    for inv in invoices:
        inv_id = inv.get("id")
        if inv_id is None or inv_id in seen_ids:
            continue
        seen_ids.add(inv_id)

        if customer_id is not None and inv.get("customer_id") != customer_id:
            continue

        if str(inv.get("status", "OPEN")).upper() != "OPEN":
            continue

        inv_curr = str(inv.get("currency") or inv.get("original_currency") or "USD").upper()
        inv_orig_val = float(inv.get("original_value") if inv.get("original_value") is not None else inv.get("value", 0.0))

        inv_rate = float(
            inv.get("exchange_rate")
            or inv.get("cotacao")
            or (exchange_rates.get(inv_curr) if exchange_rates else None)
            or (exchange_rate if inv_curr != "USD" else 1.0)
            or 1.0
        )
        if inv_rate <= 0:
            inv_rate = 1.0

        if inv_curr != "USD":
            if "value" in inv and inv.get("original_value") is not None and inv["value"] != inv_orig_val:
                val_usd = round(float(inv["value"]), 2)
            else:
                val_usd = round(inv_orig_val / inv_rate, 2)
        else:
            val_usd = round(float(inv.get("value", inv_orig_val)), 2)

        val_cents = to_cents(val_usd)
        if 0 < val_cents <= amount_cents:
            valid_invoices.append({
                "id": inv_id,
                "customer_id": inv.get("customer_id", customer_id),
                "value": val_usd,
                "currency": "USD",
                "original_value": inv_orig_val,
                "original_currency": inv_curr,
                "exchange_rate": round(inv_rate, 4),
                "due_date": str(inv.get("due_date", "2026-03-01")),
                "issue_date": str(inv.get("issue_date", "2026-01-01")),
                "status": "OPEN"
            })

    total_avail_cents = sum(to_cents(inv["value"]) for inv in valid_invoices)
    if total_avail_cents < amount_cents or not valid_invoices:
        return {
            "status": "NO_EXACT_MATCH",
            "customer_id": customer_id,
            "amount": amount_usd,
            "currency": "USD",
            "receipt_date": receipt_date,
            "exchange_rate": exchange_rate,
            "combinations_count": 0,
            "combinations": [],
            "solver_used": "None"
        }

    # 2. Execute exact subset sum solver
    solver = get_solver(
        method=solver_name,
        max_combinations=max_combinations,
        n_invoices=len(valid_invoices)
    )
    raw_combinations = solver.solve(valid_invoices, amount_cents)

    if not raw_combinations:
        return {
            "status": "NO_EXACT_MATCH",
            "customer_id": customer_id,
            "amount": amount_usd,
            "currency": "USD",
            "receipt_date": receipt_date,
            "exchange_rate": exchange_rate,
            "combinations_count": 0,
            "combinations": [],
            "solver_used": solver.__class__.__name__
        }

    # 3. Independent audit & deterministic ranking
    audited_combos = []
    ref_dt = datetime.strptime(receipt_date, "%Y-%m-%d") if receipt_date else datetime.now()

    for combo in raw_combinations:
        if not validate_combination_integrity(combo, customer_id, amount_cents):
            continue

        ids = [inv["id"] for inv in combo]
        values = [round(float(inv["value"]), 2) for inv in combo]
        total_comb = round(sum(values), 2)

        # Compute deterministic ranking keys
        overdue_days_list = []
        for inv in combo:
            try:
                due_d = datetime.strptime(str(inv.get("due_date", receipt_date)), "%Y-%m-%d")
                delay = (ref_dt - due_d).days
                if delay > 0:
                    overdue_days_list.append(delay)
            except Exception:
                pass

        overdue_count = len(overdue_days_list)
        avg_delay = (sum(overdue_days_list) / overdue_count) if overdue_count > 0 else 0
        max_val = max(values) if values else 0.0

        # Build natural language reasons
        reasons = [
            f"Exact match sum (${total_comb:,.2f} USD) with $0.00 residual balance.",
            f"Reconciles {len(combo)} invoice(s)."
        ]
        if overdue_count > 0:
            reasons.append(f"Clears {overdue_count} overdue invoice(s) (avg delay {avg_delay:.0f} days).")

        invoices_details = [
            {
                "id": inv["id"],
                "customer_id": inv.get("customer_id"),
                "value": inv["value"],
                "currency": "USD",
                "original_value": inv.get("original_value", inv["value"]),
                "original_currency": inv.get("original_currency", "USD"),
                "exchange_rate": inv.get("exchange_rate", 1.0),
                "due_date": inv.get("due_date"),
                "issue_date": inv.get("issue_date"),
                "status": "OPEN"
            }
            for inv in combo
        ]

        audited_combos.append({
            "invoice_ids": ids,
            "invoice_values": values,
            "number_of_invoices": len(combo),
            "total": total_comb,
            "currency": "USD",
            "remaining": 0.0,
            "score": round(1.0 / len(combo), 4),
            "reasons": reasons,
            "invoices_details": invoices_details,
            "_overdue_count": overdue_count,
            "_avg_delay": avg_delay,
            "_max_val": max_val
        })

    # Sort based on ranking strategy
    if ranking_strategy == "fewest_invoices":
        audited_combos.sort(key=lambda c: (c["number_of_invoices"], -c["_overdue_count"]))
    elif ranking_strategy == "overdue_first":
        audited_combos.sort(key=lambda c: (-c["_overdue_count"], -c["_avg_delay"], c["number_of_invoices"]))
    elif ranking_strategy == "highest_value_first":
        audited_combos.sort(key=lambda c: (-c["_max_val"], c["number_of_invoices"]))
    elif ranking_strategy == "oldest_first":
        audited_combos.sort(key=lambda c: (-c["_avg_delay"], c["number_of_invoices"]))

    # Final cleanup of internal sort keys and assign rank
    for rank_idx, c in enumerate(audited_combos, 1):
        c["rank"] = rank_idx
        c.pop("_overdue_count", None)
        c.pop("_avg_delay", None)
        c.pop("_max_val", None)

    return {
        "status": "EXACT_MATCH" if audited_combos else "NO_EXACT_MATCH",
        "customer_id": customer_id,
        "amount": amount_usd,
        "currency": "USD",
        "receipt_date": receipt_date,
        "exchange_rate": exchange_rate,
        "combinations_count": len(audited_combos),
        "combinations": audited_combos,
        "solver_used": solver.__class__.__name__
    }


def suggest_invoice_payments(
    customer_id: int,
    amount: float,
    invoices: List[Dict[str, Any]],
    model: Optional[Any] = None,
    payment_history: Optional[List[Dict[str, Any]]] = None,
    solver_name: str = "auto",
    max_combinations: int = 50,
    min_history_events: int = 3,
    reference_date: Optional[str] = None,
    currency: str = "USD",
    receipt_date: Optional[str] = None,
    exchange_rate: Optional[float] = None,
    exchange_rates: Optional[Dict[str, float]] = None,
    original_amount: Optional[float] = None,
    original_currency: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Core function for suggesting and ranking exact invoice payment combinations.
    Supports multi-currency invoices and receipts, normalizing all internal solving
    and total amounts to USD while preserving original currency values and exchange rates.

    Args:
        customer_id: Customer identifier.
        amount: Total received monetary amount (in USD or converted to USD).
        invoices: List of open candidate invoices (can be in foreign currencies).
        model: Trained ML/DL ranking model. If None, falls back to deterministic rules.
        payment_history: Customer historical payment events.
        solver_name: Exact algorithm ('auto', 'backtracking', 'branch_and_bound', 'dynamic_programming', 'brute_force').
        max_combinations: Maximum number of valid combinations to collect.
        min_history_events: Threshold of past payments required to activate ML.
        reference_date: Baseline date string (YYYY-MM-DD) for overdue calculation.
        currency: Currency of the received payment (default 'USD').
        receipt_date: Settlement / payment receipt date (YYYY-MM-DD).
        exchange_rate: Dollar exchange rate quotation (cotação do dólar) on the receipt date.
        exchange_rates: Optional mapping of currency codes to exchange rates against USD.
        original_amount: Optional original received amount if provided before conversion.
        original_currency: Optional original currency if different from USD.

    Returns:
        Structured response dictionary with status, customer_id, amount (USD), currency ('USD'),
        receipt_date, exchange_rate, and ranked combinations with invoice details.
    """
    # Normalize receipt date and exchange rate aliases from kwargs if present
    receipt_date = receipt_date or kwargs.get("payment_date") or kwargs.get("data_recebimento")
    exchange_rate = exchange_rate or kwargs.get("cotacao") or kwargs.get("fx_rate")

    # Currency normalization for received amount
    curr_upper = (currency or "USD").upper()
    orig_amt = original_amount
    orig_curr = original_currency or curr_upper

    if curr_upper != "USD":
        fx = exchange_rate or (exchange_rates.get(curr_upper) if exchange_rates else None) or 1.0
        if fx <= 0:
            fx = 1.0
        amount_usd = round(amount / fx, 2)
        orig_amt = orig_amt if orig_amt is not None else amount
    else:
        amount_usd = round(amount, 2)
        if exchange_rate is not None and orig_amt is None and orig_curr != "USD":
            orig_amt = round(amount_usd * exchange_rate, 2)

    if amount_usd <= 0:
        return SuggestionResponse(
            status="NO_EXACT_MATCH",
            customer_id=customer_id,
            amount=amount_usd,
            combinations=[],
            currency="USD",
            receipt_date=receipt_date,
            exchange_rate=exchange_rate,
            original_amount=orig_amt,
            original_currency=orig_curr if orig_curr != "USD" else None
        ).to_dict()

    amount_cents = to_cents(amount_usd)

    # 1. Standardize and convert invoices to USD, filter customer, open status, and value <= amount
    valid_customer_invoices = []
    for inv in invoices:
        if inv.get("customer_id") == customer_id:
            if inv.get("status", "OPEN").upper() == "OPEN":
                inv_curr = str(inv.get("currency") or inv.get("original_currency") or "USD").upper()
                inv_orig_val = float(inv.get("original_value") if inv.get("original_value") is not None else inv.get("value", 0.0))
                
                # Determine exchange rate
                inv_rate = float(
                    inv.get("exchange_rate")
                    or inv.get("cotacao")
                    or (exchange_rates.get(inv_curr) if exchange_rates else None)
                    or (exchange_rate if inv_curr != "USD" else 1.0)
                    or 1.0
                )
                if inv_rate <= 0:
                    inv_rate = 1.0

                # Convert to USD if needed
                if inv_curr != "USD":
                    # If value was already pre-converted and differs from original_value
                    if "value" in inv and inv.get("original_value") is not None and inv["value"] != inv_orig_val:
                        val_usd = round(float(inv["value"]), 2)
                    else:
                        val_usd = round(inv_orig_val / inv_rate, 2)
                else:
                    val_usd = round(float(inv.get("value", inv_orig_val)), 2)

                val_cents = to_cents(val_usd)
                if 0 < val_cents <= amount_cents:
                    inv_copy = {
                        "id": inv.get("id"),
                        "customer_id": customer_id,
                        "value": val_usd,
                        "currency": "USD",
                        "original_value": inv_orig_val,
                        "original_currency": inv_curr,
                        "exchange_rate": round(inv_rate, 4),
                        "due_date": str(inv.get("due_date", "2026-03-01")),
                        "issue_date": str(inv.get("issue_date", "2026-01-01")),
                        "status": "OPEN"
                    }
                    valid_customer_invoices.append(inv_copy)

    total_available_cents = sum(to_cents(inv["value"]) for inv in valid_customer_invoices)
    if total_available_cents < amount_cents or not valid_customer_invoices:
        return SuggestionResponse(
            status="NO_EXACT_MATCH",
            customer_id=customer_id,
            amount=amount_usd,
            combinations=[],
            currency="USD",
            receipt_date=receipt_date,
            exchange_rate=exchange_rate,
            original_amount=orig_amt,
            original_currency=orig_curr if orig_curr != "USD" else None
        ).to_dict()

    # 2. Deterministic Subset Sum Solver (operating on integer USD cents)
    solver = get_solver(
        method=solver_name,
        max_combinations=max_combinations,
        n_invoices=len(valid_customer_invoices)
    )
    raw_combinations = solver.solve(valid_customer_invoices, amount_cents)

    if not raw_combinations:
        return SuggestionResponse(
            status="NO_EXACT_MATCH",
            customer_id=customer_id,
            amount=amount_usd,
            combinations=[],
            currency="USD",
            receipt_date=receipt_date,
            exchange_rate=exchange_rate,
            original_amount=orig_amt,
            original_currency=orig_curr if orig_curr != "USD" else None
        ).to_dict()

    # 3. Feature Extraction
    feature_extractor = FeatureExtractor(reference_date=reference_date or receipt_date)
    cust_events = [p for p in (payment_history or []) if p["customer_id"] == customer_id]
    has_sufficient_history = len(cust_events) >= min_history_events
    cust_stats = feature_extractor.compute_customer_historical_stats(customer_id, payment_history)

    comb_features_list = [
        feature_extractor.extract_combination_features(
            combination=combo,
            customer_id=customer_id,
            amount=amount_usd,
            customer_stats=cust_stats
        )
        for combo in raw_combinations
    ]

    # 4. Model Scoring vs Fallback Ranking
    use_ml = (model is not None) and has_sufficient_history
    scores: List[float] = []

    if use_ml:
        try:
            df_features = pd.DataFrame(comb_features_list)
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(df_features).tolist()
            elif hasattr(model, "predict_score"):
                model_name = "Gradient Boosting" if "Gradient Boosting" in model.trained_models else list(model.trained_models.keys())[0]
                scores = model.predict_score(model_name, df_features).tolist()
            else:
                use_ml = False
        except Exception:
            use_ml = False

    if not use_ml:
        fallback_ranker = DeterministicFallbackRanker()
        scores = fallback_ranker.rank_combinations(comb_features_list)

    scores_arr = np.array(scores, dtype=float)
    if len(scores_arr) > 1 and np.max(scores_arr) > np.min(scores_arr):
        norm_scores = 0.5 + 0.45 * ((scores_arr - np.min(scores_arr)) / (np.max(scores_arr) - np.min(scores_arr)))
    else:
        norm_scores = np.clip(scores_arr, 0.5, 0.95)

    # 5. Build candidates with independent verification
    candidates: List[CombinationResult] = []
    for idx, combo in enumerate(raw_combinations):
        if not validate_combination_integrity(combo, customer_id, amount_cents):
            continue

        ids = [inv["id"] for inv in combo]
        values = [round(float(inv["value"]), 2) for inv in combo]
        total_comb = round(sum(values), 2)
        score_val = float(norm_scores[idx])

        reasons = SuggestionExplainer.explain_combination(
            features=comb_features_list[idx],
            customer_id=customer_id,
            is_fallback=not use_ml
        )

        invoices_details = [
            {
                "id": inv["id"],
                "customer_id": inv["customer_id"],
                "value": inv["value"],
                "currency": "USD",
                "original_value": inv.get("original_value", inv["value"]),
                "original_currency": inv.get("original_currency", "USD"),
                "exchange_rate": inv.get("exchange_rate", 1.0),
                "due_date": inv.get("due_date"),
                "issue_date": inv.get("issue_date"),
                "status": inv.get("status", "OPEN")
            }
            for inv in combo
        ]

        candidates.append(CombinationResult(
            invoice_ids=ids,
            invoice_values=values,
            number_of_invoices=len(combo),
            total=total_comb,
            currency="USD",
            remaining=0.0,
            score=score_val,
            rank=1,
            reasons=reasons,
            invoices_details=invoices_details
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)

    for r_idx, cand in enumerate(candidates):
        cand.rank = r_idx + 1

    return SuggestionResponse(
        status="EXACT_MATCH" if candidates else "NO_EXACT_MATCH",
        customer_id=customer_id,
        amount=amount_usd,
        combinations=candidates,
        currency="USD",
        receipt_date=receipt_date,
        exchange_rate=exchange_rate,
        original_amount=orig_amt,
        original_currency=orig_curr if orig_curr != "USD" else None
    ).to_dict()
