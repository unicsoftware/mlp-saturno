"""
Feature Engineering module for Invoices, Customer Behavioral Profiles, and Candidate Combinations.

Author: Elpidio Junior E-ABC
License: MIT
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from src.data.schemas import to_cents, to_currency, to_reais


class FeatureExtractor:
    """
    Extracts 24 structured numerical features from candidate combinations and customer profiles
    for machine learning and deep learning training and inference.
    """

    FEATURE_NAMES = [
        # Combination monetary features (Strictly based on invoice values, never on surrogate IDs)
        "comb_total_value",
        "comb_number_of_invoices",
        "comb_avg_invoice_value",
        "comb_min_invoice_value",
        "comb_max_invoice_value",
        "comb_value_spread",
        "comb_max_invoice_ratio",     # max_value / amount (percentage of total amount represented by largest invoice)
        "comb_min_invoice_ratio",     # min_value / amount (percentage of total amount represented by smallest invoice)
        "comb_avg_invoice_ratio",     # avg_value / amount
        "comb_value_std",             # Standard deviation of invoice values in the combination
        # Delinquency and age features
        "comb_overdue_count",
        "comb_overdue_ratio",
        "comb_avg_days_overdue",
        "comb_max_days_overdue",
        "comb_oldest_age",
        "comb_newest_age",
        "comb_age_spread",
        "comb_avg_age",
        # Customer historical statistics
        "cust_num_payments",
        "cust_avg_payment_amount",
        "cust_avg_invoices_per_payment",
        "cust_avg_days_overdue_hist",
        "cust_pattern_overdue_pref",
        "cust_pattern_highest_val_pref",
        "cust_pattern_oldest_pref",
        # Interaction / alignment features
        "interact_overdue_alignment",
        "interact_value_alignment",
        "interact_age_alignment",
    ]

    def __init__(self, reference_date: Optional[str] = None):
        if reference_date:
            self.reference_date = datetime.strptime(reference_date, "%Y-%m-%d")
        else:
            self.reference_date = datetime.now()

    def _get_date_diff(self, date_str: str, from_date: datetime) -> int:
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return (from_date - dt).days
        except Exception:
            return 0

    def compute_customer_historical_stats(
        self,
        customer_id: int,
        payment_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        Computes behavioral preferences and payment patterns from historical events.
        """
        if not payment_history:
            return {
                "cust_num_payments": 0.0,
                "cust_avg_payment_amount": 0.0,
                "cust_avg_invoices_per_payment": 0.0,
                "cust_avg_days_overdue_hist": 0.0,
                "cust_pattern_overdue_pref": 0.0,
                "cust_pattern_highest_val_pref": 0.0,
                "cust_pattern_oldest_pref": 0.0,
            }

        cust_events = [p for p in payment_history if p["customer_id"] == customer_id]
        if not cust_events:
            return {
                "cust_num_payments": 0.0,
                "cust_avg_payment_amount": 0.0,
                "cust_avg_invoices_per_payment": 0.0,
                "cust_avg_days_overdue_hist": 0.0,
                "cust_pattern_overdue_pref": 0.0,
                "cust_pattern_highest_val_pref": 0.0,
                "cust_pattern_oldest_pref": 0.0,
            }

        n_payments = len(cust_events)
        avg_amt = np.mean([p["amount"] for p in cust_events])
        avg_inv_count = np.mean([len(p["selected_invoice_ids"]) for p in cust_events])

        overdue_scores = []
        high_val_scores = []
        oldest_scores = []
        days_overdue_list = []

        for p in cust_events:
            p_date = datetime.strptime(p.get("payment_date", "2026-03-01"), "%Y-%m-%d")
            sel_ids = set(p["selected_invoice_ids"])
            avail = p.get("available_invoices", [])
            sel_invs = [inv for inv in avail if inv["id"] in sel_ids]

            if sel_invs:
                overdues = [
                    max(0, (p_date - datetime.strptime(inv["due_date"], "%Y-%m-%d")).days)
                    for inv in sel_invs
                ]
                days_overdue_list.extend(overdues)
                overdue_ratio = sum(1 for d in overdues if d > 0) / len(sel_invs)
                overdue_scores.append(overdue_ratio)

                if avail:
                    max_avail = max(inv["value"] for inv in avail)
                    sel_max = max(inv["value"] for inv in sel_invs)
                    high_val_scores.append(sel_max / (max_avail + 1e-5))

                ages = [
                    (p_date - datetime.strptime(inv["issue_date"], "%Y-%m-%d")).days
                    for inv in sel_invs
                ]
                oldest_scores.append(np.mean(ages) if ages else 0.0)

        pattern = cust_events[0].get("customer_pattern", "")
        pref_overdue = 1.0 if pattern == "overdue_first" else (np.mean(overdue_scores) if overdue_scores else 0.0)
        pref_high_val = 1.0 if pattern == "highest_value_first" else (np.mean(high_val_scores) if high_val_scores else 0.0)
        pref_oldest = 1.0 if pattern == "oldest_first" else ((np.mean(oldest_scores) / 100.0) if oldest_scores else 0.0)

        return {
            "cust_num_payments": float(n_payments),
            "cust_avg_payment_amount": float(avg_amt),
            "cust_avg_invoices_per_payment": float(avg_inv_count),
            "cust_avg_days_overdue_hist": float(np.mean(days_overdue_list) if days_overdue_list else 0.0),
            "cust_pattern_overdue_pref": float(pref_overdue),
            "cust_pattern_highest_val_pref": float(pref_high_val),
            "cust_pattern_oldest_pref": float(pref_oldest),
        }

    def extract_combination_features(
        self,
        combination: List[Dict[str, Any]],
        customer_id: int,
        amount: float,
        customer_stats: Optional[Dict[str, float]] = None,
        reference_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Extracts the 24 numerical feature dictionary for a specific candidate combination.
        """
        ref_date = reference_date or self.reference_date
        values = [float(inv["value"]) for inv in combination]
        total_val = sum(values)
        n_inv = len(combination)

        days_overdue_list = []
        ages_list = []

        for inv in combination:
            due_str = inv.get("due_date", "2026-03-01")
            issue_str = inv.get("issue_date", "2026-01-01")

            days_since_due = self._get_date_diff(due_str, ref_date)
            days_overdue = max(0, days_since_due)
            age = max(0, self._get_date_diff(issue_str, ref_date))

            days_overdue_list.append(days_overdue)
            ages_list.append(age)

        overdue_count = sum(1 for d in days_overdue_list if d > 0)
        overdue_ratio = overdue_count / n_inv if n_inv > 0 else 0.0
        avg_overdue = float(np.mean(days_overdue_list)) if days_overdue_list else 0.0
        max_overdue = float(np.max(days_overdue_list)) if days_overdue_list else 0.0

        oldest_age = float(np.max(ages_list)) if ages_list else 0.0
        newest_age = float(np.min(ages_list)) if ages_list else 0.0
        avg_age = float(np.mean(ages_list)) if ages_list else 0.0
        age_spread = oldest_age - newest_age

        min_val = float(np.min(values)) if values else 0.0
        max_val = float(np.max(values)) if values else 0.0
        avg_val = float(np.mean(values)) if values else 0.0
        val_spread = max_val - min_val
        val_std = float(np.std(values)) if len(values) > 1 else 0.0

        amt_safe = amount if amount > 0 else (total_val if total_val > 0 else 1.0)
        max_ratio = max_val / amt_safe
        min_ratio = min_val / amt_safe
        avg_ratio = avg_val / amt_safe

        stats = customer_stats or self.compute_customer_historical_stats(customer_id)

        pref_od = stats.get("cust_pattern_overdue_pref", 0.0)
        pref_hv = stats.get("cust_pattern_highest_val_pref", 0.0)
        pref_old = stats.get("cust_pattern_oldest_pref", 0.0)

        interact_overdue = overdue_ratio * (1.0 + pref_od * 2.0)
        interact_val = max_ratio * (1.0 + pref_hv * 2.0)
        interact_age = (avg_age / 100.0) * (1.0 + pref_old * 2.0)

        features = {
            "comb_total_value": float(total_val),
            "comb_number_of_invoices": float(n_inv),
            "comb_avg_invoice_value": avg_val,
            "comb_min_invoice_value": min_val,
            "comb_max_invoice_value": max_val,
            "comb_value_spread": val_spread,
            "comb_max_invoice_ratio": max_ratio,
            "comb_min_invoice_ratio": min_ratio,
            "comb_avg_invoice_ratio": avg_ratio,
            "comb_value_std": val_std,
            "comb_overdue_count": float(overdue_count),
            "comb_overdue_ratio": float(overdue_ratio),
            "comb_avg_days_overdue": avg_overdue,
            "comb_max_days_overdue": max_overdue,
            "comb_oldest_age": oldest_age,
            "comb_newest_age": newest_age,
            "comb_age_spread": age_spread,
            "comb_avg_age": avg_age,
            "cust_num_payments": stats.get("cust_num_payments", 0.0),
            "cust_avg_payment_amount": stats.get("cust_avg_payment_amount", 0.0),
            "cust_avg_invoices_per_payment": stats.get("cust_avg_invoices_per_payment", 0.0),
            "cust_avg_days_overdue_hist": stats.get("cust_avg_days_overdue_hist", 0.0),
            "cust_pattern_overdue_pref": pref_od,
            "cust_pattern_highest_val_pref": pref_hv,
            "cust_pattern_oldest_pref": pref_old,
            "interact_overdue_alignment": interact_overdue,
            "interact_value_alignment": interact_val,
            "interact_age_alignment": interact_age,
        }

        return features

    def create_training_dataset(
        self,
        payment_history: List[Dict[str, Any]],
        solver_func
    ) -> Tuple[pd.DataFrame, np.ndarray, List[Dict[str, Any]]]:
        """
        Creates tabular dataset (X, y) for training ML and Deep Learning models from historical events.
        """
        rows = []
        labels = []
        metadata = []

        unique_custs = set(p["customer_id"] for p in payment_history)
        cust_stats_map = {
            cid: self.compute_customer_historical_stats(cid, payment_history)
            for cid in unique_custs
        }

        for p in payment_history:
            cid = p["customer_id"]
            amt_cents = p["amount_cents"]
            amt = p["amount"]
            avail = p["available_invoices"]
            selected_ids = set(p["selected_invoice_ids"])
            p_date = datetime.strptime(p.get("payment_date", "2026-03-01"), "%Y-%m-%d")

            valid_combinations = solver_func(avail, amt_cents)
            if not valid_combinations:
                continue

            c_stats = cust_stats_map[cid]

            for combo in valid_combinations:
                combo_ids = set(inv["id"] for inv in combo)
                is_selected = (combo_ids == selected_ids)

                feats = self.extract_combination_features(
                    combination=combo,
                    customer_id=cid,
                    amount=amt,
                    customer_stats=c_stats,
                    reference_date=p_date
                )

                rows.append(feats)
                labels.append(1 if is_selected else 0)
                metadata.append({
                    "payment_id": p["payment_id"],
                    "customer_id": cid,
                    "combo_ids": list(combo_ids),
                    "is_selected": is_selected
                })

        df_X = pd.DataFrame(rows)
        y = np.array(labels, dtype=np.float32)
        return df_X, y, metadata
