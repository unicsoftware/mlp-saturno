"""
Branch and Bound exact combination solver for Subset Sum.

Author: Elpidio Junior E-ABC
License: MIT
"""

from typing import List, Dict, Any
from src.solvers.base import BaseSolver
from src.data.schemas import to_cents


class BranchAndBoundSolver(BaseSolver):
    """
    Branch and Bound Solver with bounding estimation and depth-first exploration.
    Prunes non-feasible branches dynamically based on cumulative upper bounds.
    """

    def solve(self, invoices: List[Dict[str, Any]], target_cents: int) -> List[List[Dict[str, Any]]]:
        valid_invoices = [inv for inv in invoices if to_cents(inv["value"]) <= target_cents]
        if not valid_invoices or target_cents <= 0:
            return []

        sorted_invoices = sorted(valid_invoices, key=lambda x: to_cents(x["value"]), reverse=True)
        cents_values = [to_cents(inv["value"]) for inv in sorted_invoices]
        n = len(sorted_invoices)

        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + cents_values[i]

        results = []
        stack = [(0, 0, [])]  # (current_sum, level, path)

        while stack and len(results) < self.max_combinations:
            current_sum, level, path = stack.pop()

            if current_sum == target_cents:
                results.append([sorted_invoices[i] for i in path])
                continue

            if level >= n:
                continue

            # Upper bound pruning
            if current_sum + suffix_sums[level] < target_cents:
                continue

            # Branch 1: Exclude item at current level
            stack.append((current_sum, level + 1, path))

            # Branch 2: Include item at current level
            val = cents_values[level]
            if current_sum + val <= target_cents:
                stack.append((current_sum + val, level + 1, path + [level]))

        return results
