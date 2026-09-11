"""
Dynamic Programming (DP) exact combination solver for Subset Sum.

Author: Elpidio Junior E-ABC
License: MIT
"""

from typing import List, Dict, Any
from src.solvers.base import BaseSolver
from src.data.schemas import to_cents


class DynamicProgrammingSolver(BaseSolver):
    """
    Dynamic Programming Solver (Pseudo-polynomial).
    Maintains reachable intermediate states and subsets.
    Complexity: O(N * W) where W is target_cents.
    """

    def solve(self, invoices: List[Dict[str, Any]], target_cents: int) -> List[List[Dict[str, Any]]]:
        valid_invoices = [inv for inv in invoices if to_cents(inv["value"]) <= target_cents]
        if not valid_invoices or target_cents <= 0:
            return []

        dp: Dict[int, List[List[int]]] = {0: [[]]}

        for idx, inv in enumerate(valid_invoices):
            val = to_cents(inv["value"])
            current_sums = list(dp.keys())

            for s in current_sums:
                new_sum = s + val
                if new_sum <= target_cents:
                    if new_sum not in dp:
                        dp[new_sum] = []

                    for combo in dp[s]:
                        if len(dp[new_sum]) < self.max_combinations:
                            dp[new_sum].append(combo + [idx])

        if target_cents in dp:
            return [[valid_invoices[i] for i in combo] for combo in dp[target_cents][:self.max_combinations]]
        return []
