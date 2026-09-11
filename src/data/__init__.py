"""
Data module: schemas, synthetic generation and universal data loaders (JSON, CSV, SQL).

Author: Elpidio Junior E-ABC
License: MIT
"""

from src.data.schemas import (
    Invoice,
    CustomerProfile,
    PaymentEvent,
    CombinationResult,
    SuggestionResponse,
    to_cents,
    to_currency,
    to_reais
)
from src.data.synthetic import SyntheticDataGenerator, DEFAULT_CUSTOMER_PROFILES
from src.data.loaders import UniversalDataLoader

__all__ = [
    "Invoice",
    "CustomerProfile",
    "PaymentEvent",
    "CombinationResult",
    "SuggestionResponse",
    "to_cents",
    "to_currency",
    "to_reais",
    "SyntheticDataGenerator",
    "DEFAULT_CUSTOMER_PROFILES",
    "UniversalDataLoader"
]
