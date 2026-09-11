"""
Full Inference Pipeline: Mathematical Exact Search -> 28-Feature Extraction -> PyTorch MLP Ranking.

Author: Elpidio Junior E-ABC
License: MIT
"""

import sys
import os
import json
import argparse
import joblib
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.neural_mlp import DeepLearningRankingModel, InvoiceRankingPyTorchNet
from src.data.loaders import UniversalDataLoader
from src.pipeline.inference import suggest_invoice_payments


def load_model(models_dir: str = "saved_models") -> DeepLearningRankingModel:
    """Loads saved PyTorch MLP model from disk."""
    scaler_file = os.path.join(models_dir, "pytorch_scaler.joblib")
    weights_file = os.path.join(models_dir, "pytorch_mlp_weights.pt")

    if not os.path.exists(weights_file):
        print("Model weights not found. Training model first...")
        from _mlp.train_mlp import train_models
        train_models(save_dir=models_dir)

    scaler = joblib.load(scaler_file)
    input_dim = scaler.n_features_in_

    model_net = InvoiceRankingPyTorchNet(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.2)
    model_net.load_state_dict(torch.load(weights_file, map_location="cpu"))
    model_net.eval()

    dl_wrapper = DeepLearningRankingModel()
    dl_wrapper.scaler = scaler
    dl_wrapper.model = model_net
    return dl_wrapper


def main():
    parser = argparse.ArgumentParser(description="Integrated AI Invoice Matching & PyTorch MLP Ranking Engine")
    parser.add_argument("--input", "-i", type=str, help="Path to JSON payload file (invoices, amount, customer_id, currency)")
    parser.add_argument("--csv", type=str, help="Path to CSV file with candidate invoices")
    parser.add_argument("--amount", "-a", type=float, default=None, help="Received payment amount")
    parser.add_argument("--customer", "-c", type=int, default=None, help="Customer ID")
    parser.add_argument("--currency", type=str, default="USD", help="Payment currency (USD, BRL, EUR)")
    parser.add_argument("--exchange-rate", "-fx", type=float, default=None, help="Exchange rate quotation")
    parser.add_argument("--history", type=str, default=None, help="Path to historical payments JSON file")
    parser.add_argument("--receipt-date", type=str, default=None, help="Payment receipt date (YYYY-MM-DD)")
    parser.add_argument("--solver", type=str, default="auto", help="Subset Sum solver algorithm (auto, branch_and_bound, backtracking)")
    parser.add_argument("--max-combos", type=int, default=50, help="Maximum number of combinations to collect")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output JSON path")

    args = parser.parse_args()

    print("=" * 70)
    print("🎯 INTEGRATED INFERENCE ENGINE (MATH SOLVER + MLP RANKING)")
    print("=" * 70)

    # 1. Ingest candidate open invoices
    if args.input and os.path.exists(args.input):
        print(f"Loading payload from JSON file: {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)
        amount = payload.get("amount", args.amount or 0.0)
        customer_id = payload.get("customer_id", args.customer or 41)
        currency = payload.get("currency", args.currency)
        exchange_rate = payload.get("exchange_rate", args.exchange_rate)
        receipt_date = payload.get("receipt_date", args.receipt_date or "2026-03-01")
        invoices = payload.get("invoices", [])
    elif args.csv and os.path.exists(args.csv):
        print(f"Loading invoices from CSV file: {args.csv}")
        invoices = UniversalDataLoader.load_invoices_from_csv(args.csv)
        amount = args.amount or 5500.00
        customer_id = args.customer or 42
        currency = args.currency
        exchange_rate = args.exchange_rate
        receipt_date = args.receipt_date or "2026-03-01"
    else:
        sample_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "sample_payload.json")
        if os.path.exists(sample_file):
            print(f"Loading default sample JSON: {sample_file}")
            with open(sample_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            amount = payload.get("amount", 5000.00)
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
            receipt_date = "2026-03-01"

    # 2. Load past settlement history
    history_path = args.history or os.path.join(os.path.dirname(__file__), "..", "data", "payment_history.json")
    history = []
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

    # 3. Load model (or fallback)
    try:
        model = load_model()
    except Exception as e:
        print(f"Could not load PyTorch model ({e}). Using deterministic fallback ranking.")
        model = None

    print(f"\nReconciling Received Payment of {currency} {amount:,.2f} for Customer {customer_id} on {receipt_date}...")
    result = suggest_invoice_payments(
        customer_id=customer_id,
        amount=amount,
        invoices=invoices,
        model=model,
        payment_history=history,
        solver_name=args.solver,
        max_combinations=args.max_combos,
        currency=currency,
        exchange_rate=exchange_rate,
        receipt_date=receipt_date
    )

    print(f"\nInference Result: Status = {result['status']} | Valid Subsets: {len(result['combinations'])}\n")
    for comb in result["combinations"]:
        print(f"🏆 [Rank #{comb['rank']}] Probability Score: {comb['score']:.4f}")
        print(f"   Invoices: {comb['invoice_ids']} | Face Values (USD): {comb['invoice_values']}")
        print(f"   Sum: ${comb['total']:,.2f} USD | Remainder: ${comb['remaining']:,.2f} | Count: {comb['number_of_invoices']} invoices")
        print("   Explanations:")
        for r in comb.get("reasons", []):
            print(f"     • {r}")
        print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved output JSON to: {args.output}")

    print("=" * 70)
    print("Full Output JSON:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
