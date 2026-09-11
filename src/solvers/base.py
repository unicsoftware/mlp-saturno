"""
Base interface for deterministic Subset Sum exact combination solvers.

Author: Elpidio Junior E-ABC
License: MIT
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time


class BaseSolver(ABC):
    """
    Abstract base class for exact invoice combination solvers.
    """

    def __init__(self, max_combinations: int = 100):
        self.max_combinations = max_combinations
        self.last_execution_time_ms: float = 0.0
        self.combinations_found_count: int = 0

    @abstractmethod
    def solve(self, invoices: List[Dict[str, Any]], target_cents: int) -> List[List[Dict[str, Any]]]:
        """
        Searches for subsets of invoices whose sum in exact integer cents equals target_cents.

        Args:
            invoices: List of invoice dictionaries.
            target_cents: Target received amount in integer cents.

        Returns:
            List of valid combinations (each combination is a list of invoices).
        """
        pass

    def benchmark(self, invoices: List[Dict[str, Any]], target_cents: int) -> Dict[str, Any]:
        """Executes solver and benchmarks execution latency in milliseconds."""
        start = time.perf_counter()
        results = self.solve(invoices, target_cents)
        elapsed = (time.perf_counter() - start) * 1000.0
        self.last_execution_time_ms = elapsed
        self.combinations_found_count = len(results)

        return {
            "solver_name": self.__class__.__name__,
            "execution_time_ms": round(elapsed, 4),
            "combinations_found": len(results),
            "invoices_count": len(invoices),
            "target_cents": target_cents
        }
