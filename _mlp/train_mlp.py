"""
Training script for the Deep Learning PyTorch MLP ranking model and Classical Baselines.

Author: Elpidio Junior E-ABC
License: MIT
"""

import sys
import os
import time
import json
import joblib
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic import SyntheticDataGenerator, DEFAULT_CUSTOMER_PROFILES
from src.features.engineering import FeatureExtractor
from src.solvers.solver_factory import get_solver
from src.models.baselines import BaselineModelsManager
from src.models.neural_mlp import DeepLearningRankingModel


def train_models(save_dir: str = "saved_models"):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 70)
    print("🚀 TRAINING PYTORCH MLP & BASELINE RANKING MODELS")
    print("=" * 70)

    # 1. Load or generate raw history
    history_file = "data/payment_history.json"
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        generator = SyntheticDataGenerator(seed=42)
        history = generator.generate_payment_history(payments_per_profile=35, invoices_pool_size=12)

    # 2. Extract features
    print("\n[1/3] Extracting 28 features...")
    solver = get_solver("backtracking", max_combinations=50)
    extractor = FeatureExtractor(reference_date="2026-03-01")
    df_X, y, metadata = extractor.create_training_dataset(history, solver.solve)
    print(f"✓ Feature matrix: {df_X.shape[0]} candidate combinations x {df_X.shape[1]} features.")

    # 3. Train & Evaluate Baselines
    print("\n[2/3] Training and evaluating Classical Baselines...")
    split_idx = int(len(df_X) * 0.8)
    X_train, y_train = df_X.iloc[:split_idx], y[:split_idx]
    X_test, y_test = df_X.iloc[split_idx:], y[split_idx:]
    meta_test = metadata[split_idx:]

    baselines = BaselineModelsManager(random_state=42)
    df_metrics = baselines.fit_and_evaluate(X_train, y_train, X_test, y_test, meta_test)
    print(df_metrics.to_string(index=False))

    # 4. Train PyTorch MLP
    print("\n[3/3] Training PyTorch MLP Neural Network...")
    dl_model = DeepLearningRankingModel(
        hidden_dim1=64,
        hidden_dim2=32,
        dropout_rate=0.2,
        lr=0.005,
        epochs=50,
        batch_size=32
    )
    t0 = time.perf_counter()
    dl_model.fit(df_X, y, verbose=False)
    t_train = time.perf_counter() - t0
    print(f"✓ PyTorch MLP trained in {t_train:.2f} seconds.")

    # Save artifacts
    torch_path = os.path.join(save_dir, "pytorch_mlp_weights.pt")
    scaler_path = os.path.join(save_dir, "pytorch_scaler.joblib")
    baselines_path = os.path.join(save_dir, "baseline_models.joblib")

    torch.save(dl_model.model.state_dict(), torch_path)
    joblib.dump(dl_model.scaler, scaler_path)
    joblib.dump(baselines, baselines_path)

    print(f"\n✓ Saved PyTorch weights to: '{torch_path}'")
    print(f"✓ Saved Feature Scaler to: '{scaler_path}'")
    print(f"✓ Saved Baseline Models to: '{baselines_path}'")
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    train_models()
