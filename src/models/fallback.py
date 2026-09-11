"""
Deterministic Heuristic Fallback Ranking for Cold-Start / Insufficient History scenarios.

Author: Elpidio Junior E-ABC
License: MIT
"""

from typing import List, Dict, Any
import numpy as np


class DeterministicFallbackRanker:
    """
    Deterministic heuristic ranker activated when customer has insufficient historical data.
    Configurable business priorities:
    1. Overdue invoices ratio and days of overdue;
    2. Invoice age (oldest first);
    3. Higher face values;
    4. Conciseness (fewer invoices to clear amount).
    """

    def __init__(
        self,
        weight_overdue: float = 100.0,
        weight_age: float = 10.0,
        weight_high_val: float = 5.0,
        weight_min_count: float = 2.0
    ):
        self.weight_overdue = weight_overdue
        self.weight_age = weight_age
        self.weight_high_val = weight_high_val
        self.weight_min_count = weight_min_count

    def compute_heuristic_score(self, features_dict: Dict[str, float]) -> float:
        """Computes normalized heuristic ranking score."""
        overdue_part = features_dict.get("comb_overdue_ratio", 0.0) * self.weight_overdue
        age_part = (features_dict.get("comb_avg_age", 0.0) / 100.0) * self.weight_age
        val_part = (features_dict.get("comb_max_invoice_value", 0.0) / (features_dict.get("comb_total_value", 1.0) + 1e-5)) * self.weight_high_val
        count_part = (1.0 / (features_dict.get("comb_number_of_invoices", 1.0) + 1e-5)) * self.weight_min_count

        raw_score = overdue_part + age_part + val_part + count_part
        sigmoid_score = 1.0 / (1.0 + np.exp(-raw_score / 20.0))
        return float(sigmoid_score)

    def rank_combinations(self, combinations_features: List[Dict[str, float]]) -> List[float]:
        """Returns heuristic ranking scores for a list of candidate combinations."""
        return [self.compute_heuristic_score(f) for f in combinations_features]
