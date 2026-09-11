"""
Typed data structures and schemas for invoice payment suggestion system.

Author: Elpidio Junior E-ABC
License: MIT
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import date, datetime


def to_cents(amount: float) -> int:
    """Safely converts monetary currency float value to exact integer cents."""
    return int(round(amount * 100))


def to_currency(cents: int) -> float:
    """Converts integer cents to float currency with 2 decimal places."""
    return round(cents / 100.0, 2)


# Alias for backward compatibility
to_reais = to_currency


@dataclass
class Invoice:
    id: int
    customer_id: int
    value: float
    due_date: str  # Format YYYY-MM-DD
    issue_date: str  # Format YYYY-MM-DD
    status: str = "OPEN"
    currency: str = "USD"
    original_value: Optional[float] = None
    original_currency: Optional[str] = None
    exchange_rate: float = 1.0
    value_cents: int = field(init=False)

    def __post_init__(self):
        if self.original_value is None:
            self.original_value = float(self.value)
        else:
            self.original_value = float(self.original_value)

        if self.original_currency is None:
            self.original_currency = self.currency or "USD"

        if self.exchange_rate <= 0:
            self.exchange_rate = 1.0

        # If original currency is foreign and value wasn't explicitly pre-converted to USD
        if self.original_currency.upper() != "USD" and self.value == self.original_value and self.exchange_rate != 1.0:
            self.value = round(self.original_value / self.exchange_rate, 2)
        else:
            self.value = round(float(self.value), 2)

        self.value_cents = to_cents(self.value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "value": self.value,
            "value_cents": self.value_cents,
            "currency": "USD",
            "original_value": round(self.original_value, 2) if self.original_value is not None else self.value,
            "original_currency": self.original_currency or "USD",
            "exchange_rate": round(self.exchange_rate, 4),
            "due_date": self.due_date,
            "issue_date": self.issue_date,
            "status": self.status
        }


@dataclass
class CustomerProfile:
    customer_id: int
    name: str
    behavior_pattern: str  # 'overdue_first', 'highest_value_first', 'oldest_first', 'random'
    payment_frequency_days: int = 30
    default_risk: float = 0.1


@dataclass
class PaymentEvent:
    payment_id: int
    customer_id: int
    amount: float
    amount_cents: int
    payment_date: str
    available_invoices: List[Dict[str, Any]]
    selected_invoice_ids: List[int]
    currency: str = "USD"
    exchange_rate: float = 1.0
    total_selected_cents: int = field(init=False)

    def __post_init__(self):
        self.amount = round(float(self.amount), 2)
        self.amount_cents = to_cents(self.amount)
        sel_set = set(self.selected_invoice_ids)
        self.total_selected_cents = sum(
            to_cents(inv.get("value", inv.get("amount", 0.0)))
            for inv in self.available_invoices
            if inv["id"] in sel_set
        )


@dataclass
class CombinationResult:
    invoice_ids: List[int]
    invoice_values: List[float]
    number_of_invoices: int
    total: float
    currency: str = "USD"
    remaining: float = 0.0
    score: float = 0.0
    rank: int = 1
    reasons: List[str] = field(default_factory=list)
    invoices_details: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "rank": self.rank,
            "currency": self.currency,
            "invoice_ids": self.invoice_ids,
            "invoice_values": self.invoice_values,
            "number_of_invoices": self.number_of_invoices,
            "total": round(self.total, 2),
            "remaining": round(self.remaining, 2),
            "score": round(self.score, 4)
        }
        if self.invoices_details:
            res["invoices_details"] = self.invoices_details
        if self.reasons:
            res["reasons"] = self.reasons
        return res


@dataclass
class SuggestionResponse:
    status: str  # "EXACT_MATCH" or "NO_EXACT_MATCH"
    customer_id: int
    amount: float
    combinations: List[CombinationResult]
    currency: str = "USD"
    receipt_date: Optional[str] = None
    exchange_rate: Optional[float] = None
    original_amount: Optional[float] = None
    original_currency: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "status": self.status,
            "customer_id": self.customer_id,
            "currency": self.currency,
            "amount": round(self.amount, 2),
        }
        if self.receipt_date is not None:
            res["receipt_date"] = self.receipt_date
        if self.exchange_rate is not None:
            res["exchange_rate"] = round(float(self.exchange_rate), 4)
        if self.original_amount is not None:
            res["original_amount"] = round(float(self.original_amount), 2)
        if self.original_currency is not None:
            res["original_currency"] = self.original_currency

        res["combinations"] = [c.to_dict() for c in self.combinations]
        return res
