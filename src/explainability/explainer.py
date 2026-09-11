"""
Explainability and Natural Language Justification module for suggested invoice combinations.

Author: Elpidio Junior E-ABC
License: MIT
"""

from typing import List, Dict, Any


class SuggestionExplainer:
    """
    Generates transparent explainability factors for recommended invoice combinations.
    """

    @staticmethod
    def explain_combination(
        features: Dict[str, float],
        customer_id: int,
        is_fallback: bool = False
    ) -> List[str]:
        """
        Returns list of explainability reasons for the candidate combination.
        """
        reasons = []

        # Factor 1: Exact match guarantee
        reasons.append("The combination matches the exact received amount with zero remaining balance (exact match).")

        # Factor 2: Overdue invoices
        overdue_cnt = int(features.get("comb_overdue_count", 0))
        if overdue_cnt > 0:
            avg_od = int(features.get("comb_avg_days_overdue", 0))
            reasons.append(f"Contains {overdue_cnt} overdue invoice(s) with an average delay of {avg_od} days.")

        # Factor 3: Customer historical preference alignment
        if not is_fallback:
            pref_od = features.get("cust_pattern_overdue_pref", 0.0)
            pref_hv = features.get("cust_pattern_highest_val_pref", 0.0)
            pref_old = features.get("cust_pattern_oldest_pref", 0.0)

            if pref_od >= 0.6 and overdue_cnt > 0:
                reasons.append("Customer payment history indicates strong preference for clearing overdue invoices first.")
            elif pref_hv >= 0.6:
                max_v = features.get("comb_max_invoice_value", 0.0)
                reasons.append(f"Customer historically prioritizes highest value invoices (largest in set: ${max_v:,.2f}).")
            elif pref_old >= 0.6:
                avg_age = int(features.get("comb_avg_age", 0))
                reasons.append(f"Customer historically clears oldest invoices first (average invoice age: {avg_age} days).")
        else:
            reasons.append("Ranked via standard accounting business rules (fallback for cold-start customer).")

        # Factor 4: Conciseness
        n_inv = int(features.get("comb_number_of_invoices", 1))
        if n_inv <= 2:
            reasons.append(f"Efficient reconciliation using concise subset ({n_inv} invoices).")

        return reasons
