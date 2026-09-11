"""
Optimized Backtracking solver with Suffix Sum pruning for Subset Sum.

Author: Elpidio Junior E-ABC
License: MIT
"""

from typing import List, Dict, Any
from src.solvers.base import BaseSolver
from src.data.schemas import to_cents


class BacktrackingSolver(BaseSolver):
    """
    Backtracking Solver with aggressive pruning (descending sorting and suffix sums).
    Complexity: typically orders of magnitude faster than O(2^N) in practice.
    """

    def solve(self, invoices: List[Dict[str, Any]], target_cents: int) -> List[List[Dict[str, Any]]]:
        valid_invoices = [inv for inv in invoices if to_cents(inv["value"]) <= target_cents]
        if not valid_invoices or target_cents <= 0:
            return []

        # Sort descending to explore heavy branches and prune early
        sorted_invoices = sorted(valid_invoices, key=lambda x: to_cents(x["value"]), reverse=True)
        cents_values = [to_cents(inv["value"]) for inv in sorted_invoices]
        n = len(sorted_invoices)

        # Precompute suffix sums for feasibility pruning
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + cents_values[i]

        results: List[List[Dict[str, Any]]] = []

        def backtrack(idx: int, current_sum: int, path: List[int]):
            if len(results) >= self.max_combinations:
                return

            if current_sum == target_cents:
                results.append([sorted_invoices[i] for i in path])
                return

            if idx >= n:
                return

            # Pruning 1: If current sum + all remaining values < target, impossible to reach target
            if current_sum + suffix_sums[idx] < target_cents:
                return

            val = cents_values[idx]

            # Branch 1: Include current item (if it fits)
            if current_sum + val <= target_cents:
                backtrack(idx + 1, current_sum + val, path + [idx])

            # Branch 2: Exclude current item
            backtrack(idx + 1, current_sum, path)

        backtrack(0, 0, [])
        return results
