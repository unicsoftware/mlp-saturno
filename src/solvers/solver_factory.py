"""
Factory and solver comparison utilities for Subset Sum.

Author: Elpidio Junior E-ABC
License: MIT
"""

from typing import List, Dict, Any, Type
import pandas as pd
from src.solvers.base import BaseSolver
from src.solvers.brute_force import BruteForceSolver
from src.solvers.backtracking import BacktrackingSolver
from src.solvers.dynamic_programming import DynamicProgrammingSolver
from src.solvers.branch_and_bound import BranchAndBoundSolver


SOLVER_REGISTRY: Dict[str, Type[BaseSolver]] = {
    "brute_force": BruteForceSolver,
    "backtracking": BacktrackingSolver,
    "dynamic_programming": DynamicProgrammingSolver,
    "branch_and_bound": BranchAndBoundSolver
}


def get_solver(method: str = "auto", max_combinations: int = 100, n_invoices: int = 10) -> BaseSolver:
    """
    Returns requested solver instance or automatically selects optimal algorithm.

    Args:
        method: Solver name ('auto', 'backtracking', 'branch_and_bound', 'dynamic_programming', 'brute_force').
        max_combinations: Maximum number of combinations to collect.
        n_invoices: Number of candidate invoices (used in 'auto' decision).

    Returns:
        Instance of BaseSolver.
    """
    if method == "auto":
        if n_invoices <= 12:
            return BacktrackingSolver(max_combinations=max_combinations)
        elif n_invoices <= 30:
            return BacktrackingSolver(max_combinations=max_combinations)
        else:
            return BranchAndBoundSolver(max_combinations=max_combinations)

    method_key = method.lower().replace(" ", "_").replace("-", "_")
    if method_key not in SOLVER_REGISTRY:
        raise ValueError(f"Unknown solver: '{method}'. Available options: {list(SOLVER_REGISTRY.keys()) + ['auto']}")

    return SOLVER_REGISTRY[method_key](max_combinations=max_combinations)


def compare_solvers(
    invoices: List[Dict[str, Any]],
    target_cents: int,
    max_combinations: int = 100,
    include_brute_force: bool = True
) -> pd.DataFrame:
    """
    Executes all available solvers on the same invoice set and target, comparing benchmark metrics.

    Returns:
        DataFrame with latency, combination counts and parameters.
    """
    rows = []
    solvers = [
        ("Backtracking", BacktrackingSolver(max_combinations=max_combinations)),
        ("Branch and Bound", BranchAndBoundSolver(max_combinations=max_combinations)),
        ("Dynamic Programming", DynamicProgrammingSolver(max_combinations=max_combinations)),
    ]

    if include_brute_force and len(invoices) <= 18:
        solvers.insert(0, ("Brute Force", BruteForceSolver(max_combinations=max_combinations)))

    for name, solver in solvers:
        bench = solver.benchmark(invoices, target_cents)
        rows.append({
            "Algorithm": name,
            "Latency (ms)": bench["execution_time_ms"],
            "Combinations Found": bench["combinations_found"],
            "N Invoices": bench["invoices_count"],
            "Target (Cents)": bench["target_cents"]
        })

    return pd.DataFrame(rows)
