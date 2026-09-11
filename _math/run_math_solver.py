"""
Pure Mathematical Engine: Deterministic Subset Sum Solver.
Finds 100% of exact invoice combinations without requiring Machine Learning.

Author: Elpidio Junior E-ABC
License: MIT
"""

import sys
import os
import json
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.schemas import to_cents, to_currency
from src.data.loaders import UniversalDataLoader
from src.solvers.solver_factory import get_solver


def run_pure_math_solver(
    customer_id: int,
    amount: float,
    invoices: List[Dict[str, Any]],
    solver_name: str = "auto",
    max_combinations: int = 50
) -> Dict[str, Any]:
    """
    Executes purely the deterministic mathematical engine.
    Filters invoices by customer_id and finds exact integer cents combinations.
    """
    amount_cents = to_cents(amount)

    # 1. Filter customer invoices
    valid_invoices = [
        inv for inv in invoices
        if inv.get("customer_id") == customer_id
        and str(inv.get("status", "OPEN")).upper() == "OPEN"
        and 0 < to_cents(float(inv.get("value", 0.0))) <= amount_cents
    ]

    total_avail_cents = sum(to_cents(inv["value"]) for inv in valid_invoices)
    if total_avail_cents < amount_cents or not valid_invoices:
        return {
            "status": "NO_EXACT_MATCH",
            "customer_id": customer_id,
            "amount": amount,
            "combinations_count": 0,
            "combinations": []
        }

    # 2. Execute mathematical solver
    solver = get_solver(solver_name, max_combinations=max_combinations, n_invoices=len(valid_invoices))
    raw_subsets = solver.solve(valid_invoices, amount_cents)

    if not raw_subsets:
        return {
            "status": "NO_EXACT_MATCH",
            "customer_id": customer_id,
            "amount": amount,
            "combinations_count": 0,
            "combinations": []
        }

    # 3. Format mathematical results
    formatted_combinations = []
    for idx, combo in enumerate(raw_subsets, 1):
        ids = [inv["id"] for inv in combo]
        values = [round(float(inv["value"]), 2) for inv in combo]
        total = round(sum(values), 2)
        formatted_combinations.append({
            "combination_number": idx,
            "invoice_ids": ids,
            "invoice_values": values,
            "number_of_invoices": len(combo),
            "total": total,
            "remaining": 0.0
        })

    return {
        "status": "EXACT_MATCH",
        "customer_id": customer_id,
        "amount": amount,
        "solver_used": solver.__class__.__name__,
        "combinations_count": len(formatted_combinations),
        "combinations": formatted_combinations
    }


def main():
    print("=" * 70)
    print("🔢 PURE MATHEMATICAL ENGINE - EXACT SUBSET SUM SOLVER")
    print("=" * 70)

    sample_json = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "my_invoices.json")
    if os.path.exists(sample_json):
        invoices = UniversalDataLoader.load_invoices_from_json(sample_json)
    else:
        invoices = [
            {"id": 1001, "customer_id": 41, "value": 2000.00, "status": "OPEN"},
            {"id": 1002, "customer_id": 41, "value": 3500.00, "status": "OPEN"},
            {"id": 1003, "customer_id": 41, "value": 4500.00, "status": "OPEN"},
            {"id": 1004, "customer_id": 41, "value": 5000.00, "status": "OPEN"},
            {"id": 1005, "customer_id": 41, "value": 5000.00, "status": "OPEN"},
            {"id": 1006, "customer_id": 41, "value": 1500.00, "status": "OPEN"},
        ]

    customer_id = 41
    amount = 10000.00

    print(f"\nSearching exact mathematical combinations for Customer {customer_id} with Amount ${amount:,.2f}...")
    res = run_pure_math_solver(customer_id, amount, invoices, solver_name="auto")

    print(f"\nResult: Status = {res['status']} | Combinations Found: {res['combinations_count']} (Solver: {res.get('solver_used', 'N/A')})\n")
    for c in res["combinations"]:
        print(f"✓ Combination #{c['combination_number']}: IDs {c['invoice_ids']} | Values: {c['invoice_values']} | Sum: ${c['total']:,.2f} | Remaining: ${c['remaining']:,.2f}")

    print("\n" + "=" * 70)
    print("JSON Output:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
