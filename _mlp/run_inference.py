"""
Full Inference Pipeline: Mathematical Exact Search -> 28-Feature Extraction -> PyTorch MLP Ranking.

Author: Elpidio Junior E-ABC
License: MIT
"""

import sys
import os
import json
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
    print("=" * 70)
    print("🎯 INTEGRATED INFERENCE ENGINE (MATH SOLVER + MLP RANKING)")
    print("=" * 70)

    # 1. Load trained PyTorch Ranking Model
    model = load_model()

    # 2. Ingest candidate open invoices
    sample_file = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "my_invoices.json")
    invoices = UniversalDataLoader.load_invoices_from_json(sample_file)

    customer_id = 41
    amount = 10000.00

    # 3. Load past settlement history for customer behavioral profiling
    history_file = os.path.join(os.path.dirname(__file__), "..", "data", "payment_history.json")
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    print(f"\nReconciling Received Payment of ${amount:,.2f} for Customer {customer_id}...")
    result = suggest_invoice_payments(
        customer_id=customer_id,
        amount=amount,
        invoices=invoices,
        model=model,
        payment_history=history,
        reference_date="2026-03-01"
    )

    print(f"\nInference Result: Status = {result['status']} | Valid Subsets: {len(result['combinations'])}\n")
    for comb in result["combinations"]:
        print(f"🏆 [Rank #{comb['rank']}] Probability Score: {comb['score']:.4f}")
        print(f"   Invoices: {comb['invoice_ids']} | Face Values: {comb['invoice_values']}")
        print(f"   Sum: ${comb['total']:,.2f} | Remainder: ${comb['remaining']:,.2f} | Count: {comb['number_of_invoices']} invoices")
        print("   Explanations:")
        for r in comb.get("reasons", []):
            print(f"     • {r}")
        print()

    print("=" * 70)
    print("Full Output JSON:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
