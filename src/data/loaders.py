"""
Universal data loaders to ingest real-world invoices and settlement history from JSON, CSV, and SQL Databases.

Author: Elpidio Junior E-ABC
License: MIT
"""

import json
from typing import List, Dict, Any, Union, Optional
import pandas as pd

from src.data.schemas import to_cents, to_currency


class UniversalDataLoader:
    """
    Standardizes loading and conversion of real invoices and payment history
    from JSON files, CSV/Excel spreadsheets, and SQL databases.
    """

    @staticmethod
    def load_invoices_from_json(json_path_or_str: str) -> List[Dict[str, Any]]:
        """Loads candidate invoices list from a JSON file or JSON string."""
        if json_path_or_str.strip().startswith("[") or json_path_or_str.strip().startswith("{"):
            raw_data = json.loads(json_path_or_str)
        else:
            with open(json_path_or_str, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

        if isinstance(raw_data, dict):
            raw_data = raw_data.get("invoices", [raw_data])

        standardized = []
        for item in raw_data:
            curr = str(item.get("currency") or item.get("original_currency") or "USD").upper()
            val = float(item.get("value") or item.get("amount") or item.get("gross_amount") or 0.0)
            orig_val = float(item.get("original_value") if item.get("original_value") is not None else val)
            rate = float(item.get("exchange_rate") or item.get("cotacao") or 1.0)
            
            standardized.append({
                "id": item.get("id") or item.get("invoice_id"),
                "customer_id": int(item.get("customer_id")),
                "value": val,
                "currency": "USD" if curr == "USD" else curr,
                "original_value": orig_val,
                "original_currency": curr,
                "exchange_rate": rate,
                "due_date": str(item.get("due_date", "2026-03-01")),
                "issue_date": str(item.get("issue_date", "2026-01-01")),
                "status": str(item.get("status", "OPEN")).upper()
            })
        return standardized

    @staticmethod
    def load_invoices_from_csv(csv_path: str) -> List[Dict[str, Any]]:
        """Loads and standardizes candidate open invoices from a CSV spreadsheet."""
        df = pd.read_csv(csv_path)
        # Column name aliases normalization
        col_map = {
            "invoice_id": "id",
            "gross_amount": "value",
            "invoice_amount": "value",
            "amount": "value",
            "client_id": "customer_id",
            "dt_vencimento": "due_date",
            "dt_emissao": "issue_date",
            "moeda": "currency",
            "cotacao": "exchange_rate",
            "fx_rate": "exchange_rate"
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        invoices = []
        for _, row in df.iterrows():
            curr = str(row.get("currency") or "USD").upper()
            val = float(row.get("value", 0.0))
            orig_val = float(row.get("original_value", val))
            rate = float(row.get("exchange_rate", 1.0))

            invoices.append({
                "id": row.get("id"),
                "customer_id": int(row.get("customer_id")),
                "value": val,
                "currency": "USD" if curr == "USD" else curr,
                "original_value": orig_val,
                "original_currency": curr,
                "exchange_rate": rate,
                "due_date": str(row.get("due_date", "2026-03-01")),
                "issue_date": str(row.get("issue_date", "2026-01-01")),
                "status": str(row.get("status", "OPEN")).upper()
            })
        return invoices

    @staticmethod
    def load_payment_history_from_json(json_path: str) -> List[Dict[str, Any]]:
        """Loads historical settlement events from a JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            raw_history = json.load(f)

        standardized = []
        for event in raw_history:
            amt = float(event.get("amount") or event.get("amount_received"))
            curr = str(event.get("currency") or "USD").upper()
            rate = float(event.get("exchange_rate") or event.get("cotacao") or 1.0)
            standardized.append({
                "payment_id": event.get("payment_id") or event.get("id"),
                "customer_id": int(event.get("customer_id")),
                "amount": amt,
                "amount_cents": to_cents(amt),
                "currency": curr,
                "exchange_rate": rate,
                "payment_date": str(event.get("payment_date") or event.get("receipt_date", "2026-03-01")),
                "available_invoices": event.get("available_invoices", []),
                "selected_invoice_ids": list(event.get("selected_invoice_ids", []))
            })
        return standardized

    @staticmethod
    def load_payment_history_from_sql(
        db_connection_uri: str,
        events_table_or_query: str = "vw_training_payment_history",
        invoices_table: str = "erp_invoices"
    ) -> List[Dict[str, Any]]:
        """
        Queries and builds historical payment dataset directly from an SQL relational database
        (PostgreSQL, MySQL, Oracle, SQL Server, SQLite) using SQLAlchemy.
        """
        from sqlalchemy import create_engine
        engine = create_engine(db_connection_uri)

        # Query settlement events
        if "select" in events_table_or_query.lower():
            df_events = pd.read_sql(events_table_or_query, engine)
        else:
            df_events = pd.read_sql(f"SELECT * FROM {events_table_or_query}", engine)

        # Query all open/historical invoices
        df_all_invs = pd.read_sql(f"SELECT * FROM {invoices_table}", engine)
        invs_by_cust: Dict[int, list] = {}
        for _, r in df_all_invs.iterrows():
            cid = int(r["customer_id"])
            if cid not in invs_by_cust:
                invs_by_cust[cid] = []
            curr = str(r.get("currency") or "USD").upper()
            val = float(r.get("value") or r.get("gross_amount") or 0.0)
            orig_val = float(r.get("original_value", val))
            rate = float(r.get("exchange_rate") or r.get("cotacao") or 1.0)

            invs_by_cust[cid].append({
                "id": r.get("id"),
                "customer_id": cid,
                "value": val,
                "currency": curr,
                "original_value": orig_val,
                "original_currency": curr,
                "exchange_rate": rate,
                "due_date": str(r.get("due_date")),
                "issue_date": str(r.get("issue_date")),
                "status": str(r.get("status", "OPEN")).upper()
            })

        history = []
        for _, ev in df_events.iterrows():
            cid = int(ev["customer_id"])
            amt = float(ev.get("amount") or ev.get("amount_received"))
            curr = str(ev.get("currency") or "USD").upper()
            rate = float(ev.get("exchange_rate") or ev.get("cotacao") or 1.0)
            
            # Parse selected IDs if stored as comma-separated or list
            sel_ids = ev.get("selected_invoice_ids")
            if isinstance(sel_ids, str):
                sel_ids = [int(x.strip()) for x in sel_ids.replace("{", "").replace("}", "").split(",") if x.strip()]

            history.append({
                "payment_id": ev.get("payment_id"),
                "customer_id": cid,
                "amount": amt,
                "amount_cents": to_cents(amt),
                "currency": curr,
                "exchange_rate": rate,
                "payment_date": str(ev.get("payment_date") or ev.get("receipt_date", "2026-03-01")),
                "available_invoices": invs_by_cust.get(cid, []),
                "selected_invoice_ids": list(sel_ids or [])
            })

        return history
