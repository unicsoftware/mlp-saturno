"""
Pure Mathematical Engine: Deterministic Subset Sum Solver.
Finds 100% of exact invoice combinations without requiring Machine Learning.

Can be executed standalone, imported by external systems/platforms, or run via CLI with JSON/CSV.

Author: Elpidio Junior E-ABC
License: MIT
"""

import sys
import os
import json
import argparse
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.schemas import to_cents, to_currency
from src.data.loaders import UniversalDataLoader
from src.pipeline.inference import find_deterministic_matches


def run_pure_math_solver(
    customer_id: Optional[int],
    amount: float,
    invoices: List[Dict[str, Any]],
    currency: str = "USD",
    exchange_rate: Optional[float] = None,
    receipt_date: Optional[str] = None,
    solver_name: str = "auto",
    max_combinations: int = 50,
    ranking_strategy: str = "fewest_invoices"
) -> Dict[str, Any]:
    """
    Executes purely the deterministic mathematical engine with multi-currency USD normalization.
    """
    return find_deterministic_matches(
        amount=amount,
        invoices=invoices,
        customer_id=customer_id,
        currency=currency,
        exchange_rate=exchange_rate,
        receipt_date=receipt_date,
        solver_name=solver_name,
        max_combinations=max_combinations,
        ranking_strategy=ranking_strategy
    )


def main():
    parser = argparse.ArgumentParser(description="Deterministic Mathematical Invoice Matching Engine")
    parser.add_argument("--input", "-i", type=str, help="Path to JSON payload file (containing amount, invoices, currency, etc.)")
    parser.add_argument("--csv", type=str, help="Path to CSV file with candidate invoices")
    parser.add_argument("--amount", "-a", type=float, default=None, help="Received payment amount")
    parser.add_argument("--customer", "-c", type=int, default=None, help="Customer ID (optional)")
    parser.add_argument("--currency", type=str, default="USD", help="Payment currency (default USD)")
    parser.add_argument("--exchange-rate", "-fx", type=float, default=None, help="Dollar exchange rate quotation")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output JSON path")

    args = parser.parse_args()

    print("=" * 70)
    print("🔢 PURE MATHEMATICAL ENGINE - DETERMINISTIC SUBSET SUM SOLVER")
    print("=" * 70)

    if args.input and os.path.exists(args.input):
        print(f"Loading payload from JSON file: {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)
        amount = payload.get("amount", args.amount or 0.0)
        customer_id = payload.get("customer_id", args.customer)
        currency = payload.get("currency", args.currency)
        exchange_rate = payload.get("exchange_rate", args.exchange_rate)
        receipt_date = payload.get("receipt_date")
        invoices = payload.get("invoices", [])
    elif args.csv and os.path.exists(args.csv):
        print(f"Loading invoices from CSV file: {args.csv}")
        invoices = UniversalDataLoader.load_invoices_from_csv(args.csv)
        amount = args.amount or 5000.00
        customer_id = args.customer
        currency = args.currency
        exchange_rate = args.exchange_rate
        receipt_date = None
    else:
        sample_json = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "sample_payload.json")
        if os.path.exists(sample_json):
            print(f"Using default sample payload: {sample_json}")
            with open(sample_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            amount = payload.get("amount", 5000.0)
            customer_id = payload.get("customer_id", 41)
            currency = payload.get("currency", "USD")
            exchange_rate = payload.get("exchange_rate", 5.0)
            receipt_date = payload.get("receipt_date", "2026-03-15")
            invoices = payload.get("invoices", [])
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
            currency = "USD"
            exchange_rate = 1.0
            receipt_date = None

    print(f"\nSearching exact mathematical combinations for Customer {customer_id} with Amount {currency} {amount:,.2f}...")
    res = run_pure_math_solver(
        customer_id=customer_id,
        amount=amount,
        invoices=invoices,
        currency=currency,
        exchange_rate=exchange_rate,
        receipt_date=receipt_date
    )

    print(f"\nResult: Status = {res['status']} | Combinations Found: {res['combinations_count']} (Solver: {res.get('solver_used', 'N/A')})\n")
    for c in res["combinations"]:
        print(f"✓ Rank #{c['rank']}: IDs {c['invoice_ids']} | Values: {c['invoice_values']} | Sum: ${c['total']:,.2f} | Remaining: ${c['remaining']:,.2f}")
        for r in c.get("reasons", []):
            print(f"    • {r}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved output JSON to: {args.output}")

    print("\n" + "=" * 70)
    print("JSON Output Preview:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
