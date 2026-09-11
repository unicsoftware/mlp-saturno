"""
Mathematical Solvers Benchmark Comparison.
Compares Brute Force, Backtracking, Dynamic Programming, and Branch & Bound across batch sizes.

Author: Elpidio Junior E-ABC
License: MIT
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.schemas import to_cents
from src.data.synthetic import SyntheticDataGenerator
from src.solvers.solver_factory import compare_solvers


def run_benchmarks():
    print("=" * 70)
    print("⏱️ MATHEMATICAL SOLVERS BENCHMARK (SUBSET SUM)")
    print("=" * 70)

    generator = SyntheticDataGenerator(seed=42)
    bench_scenarios = generator.generate_benchmark_datasets()

    results = []
    for n_size in [5, 10, 20, 50, 100]:
        amt, invs = bench_scenarios[n_size]
        amt_cents = to_cents(amt)
        print(f"\nEvaluating N={n_size} invoices (Target Amount: ${amt:,.2f})...")

        df_res = compare_solvers(
            invoices=invs,
            target_cents=amt_cents,
            max_combinations=50,
            include_brute_force=(n_size <= 10)
        )
        df_res["N Invoices"] = n_size
        results.append(df_res)

    df_all = pd.concat(results, ignore_index=True)
    pivot = df_all.pivot_table(index="N Invoices", columns="Algorithm", values="Latency (ms)")

    print("\n" + "=" * 70)
    print("📊 Latency Comparison Matrix (in milliseconds):")
    print("=" * 70)
    print(pivot.to_string())
    print("\n" + "=" * 70)
    print("✅ BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
