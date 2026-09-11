"""
Feature Engineering Script for Machine Learning and Deep Learning models.
Extracts 28 numerical features from raw settlement history and exports CSV datasets.

Author: Elpidio Junior E-ABC
License: MIT
"""

import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.synthetic import SyntheticDataGenerator, DEFAULT_CUSTOMER_PROFILES
from src.features.engineering import FeatureExtractor
from src.solvers.solver_factory import get_solver


def main(output_csv: str = "data/training_features_dataset.csv"):
    os.makedirs("data", exist_ok=True)
    print("=" * 70)
    print("🧠 FEATURE ENGINEERING PIPELINE (28 DIMENSIONS)")
    print("=" * 70)

    # 1. Load or generate raw history
    history_file = "data/payment_history.json"
    if os.path.exists(history_file):
        print(f"Loading existing historical settlements from '{history_file}'...")
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        print("Generating historical settlements from synthetic profiles...")
        generator = SyntheticDataGenerator(seed=42)
        history = generator.generate_payment_history(
            profiles=DEFAULT_CUSTOMER_PROFILES,
            payments_per_profile=35,
            invoices_pool_size=12
        )
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"✓ Loaded {len(history)} settlement events.")

    # 2. Extract features
    print("\nExtracting 28 numerical features (Monetary values, ratios, delinquency & interactions)...")
    solver = get_solver("backtracking", max_combinations=50)
    extractor = FeatureExtractor(reference_date="2026-03-01")

    df_X, y, metadata = extractor.create_training_dataset(history, solver.solve)

    # 3. Export CSV
    df_export = df_X.copy()
    df_export["target_is_selected"] = y.astype(int)
    df_export["payment_id"] = [m["payment_id"] for m in metadata]
    df_export["customer_id"] = [m["customer_id"] for m in metadata]

    df_export.to_csv(output_csv, index=False)
    print(f"\n✓ Exported feature matrix to '{output_csv}' ({df_export.shape[0]} rows x {df_export.shape[1]} columns)")
    print("\nFeature Columns Generated:")
    for c in df_X.columns:
        print(f"  • {c}")
    print("\n" + "=" * 70)
    print("✅ FEATURE EXTRACTION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
