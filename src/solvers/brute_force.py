"""
Brute Force exact combination solver for Subset Sum.

Author: Elpidio Junior E-ABC
License: MIT
"""

import itertools
from typing import List, Dict, Any
from src.solvers.base import BaseSolver
from src.data.schemas import to_cents


class BruteForceSolver(BaseSolver):
    """
    Brute Force Solver: iterates over combinations of sizes 1 to N.
    Complexity: O(2^N). Suitable for small N (<= 12).
    """

    def solve(self, invoices: List[Dict[str, Any]], target_cents: int) -> List[List[Dict[str, Any]]]:
        valid_invoices = [inv for inv in invoices if to_cents(inv["value"]) <= target_cents]
        n = len(valid_invoices)
        results = []

        if n == 0 or target_cents <= 0:
            return results

        for k in range(1, n + 1):
            for combo in itertools.combinations(valid_invoices, k):
                total_cents = sum(to_cents(inv["value"]) for inv in combo)
                if total_cents == target_cents:
                    results.append(list(combo))
                    if len(results) >= self.max_combinations:
                        return results

        return results
