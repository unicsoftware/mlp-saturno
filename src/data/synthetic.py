"""
Synthetic dataset generator with realistic behavioral customer profiles for accounts receivable reconciliation.

Author: Elpidio Junior E-ABC
License: MIT
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.data.schemas import Invoice, CustomerProfile, PaymentEvent, to_cents, to_currency, to_reais


# Default customer profiles for simulation, testing and learning
DEFAULT_CUSTOMER_PROFILES = [
    CustomerProfile(
        customer_id=41,
        name="Customer Alpha (Prioritizes Overdue Invoices)",
        behavior_pattern="overdue_first"
    ),
    CustomerProfile(
        customer_id=52,
        name="Customer Beta (Prioritizes Highest Face Value)",
        behavior_pattern="highest_value_first"
    ),
    CustomerProfile(
        customer_id=73,
        name="Customer Gamma (Prioritizes Oldest Invoices)",
        behavior_pattern="oldest_first"
    ),
    CustomerProfile(
        customer_id=99,
        name="Customer Delta (Arbitrary / Random Pattern)",
        behavior_pattern="random"
    ),
    CustomerProfile(
        customer_id=105,
        name="New Customer (Cold Start / Insufficient History)",
        behavior_pattern="overdue_first"
    )
]


class SyntheticDataGenerator:
    """
    Synthetic data generator simulating financial accounts receivable ecosystems.
    """

    def __init__(self, seed: int = 42, base_date: str = "2026-03-01"):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.base_date = datetime.strptime(base_date, "%Y-%m-%d")
        self.current_invoice_id = 1000
        self.current_payment_id = 5000

    def generate_invoices_for_customer(
        self,
        customer_id: int,
        n_invoices: int = 10,
        min_value: float = 100.0,
        max_value: float = 5000.0,
        step: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Generates a list of open invoices with varied issue and due dates.
        """
        invoices = []
        for _ in range(n_invoices):
            self.current_invoice_id += 1
            steps_count = int((max_value - min_value) / step)
            val = round(min_value + random.randint(0, steps_count) * step, 2)

            issue_offset = random.randint(10, 120)
            issue_dt = self.base_date - timedelta(days=issue_offset)

            due_offset = random.randint(-40, 30)
            due_dt = self.base_date + timedelta(days=due_offset)

            inv = {
                "id": self.current_invoice_id,
                "customer_id": customer_id,
                "value": val,
                "issue_date": issue_dt.strftime("%Y-%m-%d"),
                "due_date": due_dt.strftime("%Y-%m-%d"),
                "status": "OPEN"
            }
            invoices.append(inv)

        return invoices

    def _select_combination_by_profile(
        self,
        profile: CustomerProfile,
        valid_combinations: List[List[Dict[str, Any]]],
        reference_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Selects the combination that the customer would historically choose according to their profile.
        """
        if not valid_combinations:
            return []
        if len(valid_combinations) == 1:
            return valid_combinations[0]

        pattern = profile.behavior_pattern

        def score_combination(comb: List[Dict[str, Any]]) -> float:
            overdue_count = 0
            total_days_overdue = 0
            total_age = 0
            max_val = max(c["value"] for c in comb)
            min_val = min(c["value"] for c in comb)
            avg_val = sum(c["value"] for c in comb) / len(comb)
            n_inv = len(comb)

            for inv in comb:
                due_dt = datetime.strptime(inv["due_date"], "%Y-%m-%d")
                issue_dt = datetime.strptime(inv["issue_date"], "%Y-%m-%d")
                days_overdue = max(0, (reference_date - due_dt).days)
                age = (reference_date - issue_dt).days
                if days_overdue > 0:
                    overdue_count += 1
                    total_days_overdue += days_overdue
                total_age += age

            if pattern == "overdue_first":
                return overdue_count * 10000 + total_days_overdue * 10 - n_inv
            elif pattern == "highest_value_first":
                return max_val * 10 + avg_val * 5 - n_inv * 100
            elif pattern == "oldest_first":
                return total_age * 10 - n_inv
            else:  # random
                return random.random()

        sorted_combinations = sorted(valid_combinations, key=score_combination, reverse=True)
        return sorted_combinations[0]

    def generate_payment_history(
        self,
        profiles: List[CustomerProfile] = DEFAULT_CUSTOMER_PROFILES,
        payments_per_profile: int = 40,
        invoices_pool_size: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Generates simulated historical payments with ground truth selections for model training.
        """
        history = []

        for profile in profiles:
            n_payments = 2 if profile.customer_id == 105 else payments_per_profile

            for _ in range(n_payments):
                self.current_payment_id += 1
                pool = self.generate_invoices_for_customer(
                    customer_id=profile.customer_id,
                    n_invoices=random.randint(6, invoices_pool_size),
                    min_value=100.0,
                    max_value=3000.0,
                    step=100.0
                )

                k = min(len(pool), random.randint(2, min(4, len(pool))))
                target_subset = random.sample(pool, k)
                target_amount_cents = sum(to_cents(inv["value"]) for inv in target_subset)
                target_amount = to_currency(target_amount_cents)

                all_valid = self._find_subsets_exact(pool, target_amount_cents)

                pay_date_dt = self.base_date - timedelta(days=random.randint(5, 300))
                selected = self._select_combination_by_profile(profile, all_valid, pay_date_dt)
                selected_ids = [inv["id"] for inv in selected]

                history.append({
                    "payment_id": self.current_payment_id,
                    "customer_id": profile.customer_id,
                    "customer_pattern": profile.behavior_pattern,
                    "amount": target_amount,
                    "amount_cents": target_amount_cents,
                    "payment_date": pay_date_dt.strftime("%Y-%m-%d"),
                    "available_invoices": pool,
                    "selected_invoice_ids": selected_ids,
                    "all_valid_combinations_count": len(all_valid)
                })

        return history

    def _find_subsets_exact(self, invoices: List[Dict[str, Any]], target_cents: int) -> List[List[Dict[str, Any]]]:
        """Simple exact subset solver for training dataset construction."""
        results = []
        n = len(invoices)
        cents_values = [to_cents(inv["value"]) for inv in invoices]

        def backtrack(idx: int, current_sum: int, current_subset: List[int]):
            if current_sum == target_cents:
                results.append([invoices[i] for i in current_subset])
                return
            if idx >= n or current_sum > target_cents:
                return

            backtrack(idx + 1, current_sum + cents_values[idx], current_subset + [idx])
            backtrack(idx + 1, current_sum, current_subset)

        backtrack(0, 0, [])
        return results

    def generate_benchmark_datasets(self) -> Dict[int, Tuple[float, List[Dict[str, Any]]]]:
        """
        Generates controlled benchmarks for N = 5, 10, 20, 50, 100 invoices.
        """
        scenarios = {}
        for n in [5, 10, 20, 50, 100]:
            invs = self.generate_invoices_for_customer(
                customer_id=41,
                n_invoices=n,
                min_value=100.0,
                max_value=2000.0,
                step=50.0
            )
            sample_invs = random.sample(invs, min(3, len(invs)))
            amt_cents = sum(to_cents(x["value"]) for x in sample_invs)
            amt = to_currency(amt_cents)
            scenarios[n] = (amt, invs)
        return scenarios
