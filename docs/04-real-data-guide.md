# Guide for Integration with Real ERP / Accounts Receivable Data

**Author**: Elpidio Junior E-ABC  
**License**: MIT

This guide describes how to connect and ingest real invoice datasets and historical payment logs from enterprise ERP databases (e.g. SAP, Oracle EBS, Microsoft Dynamics, NetSuite, TOTVS, PostgreSQL, MySQL, SQL Server) into the AI Payment Suggestion pipeline.

---

## 1. Field Specifications & Supported Data Formats

### 1.1 Invoices Dataset (`my_invoices.json` / CSV)
Represents candidate open invoices available in the ERP system for settlement.

| Field Name | Data Type | Required? | Description & Role in Engine | Example |
| :--- | :--- | :---: | :--- | :--- |
| `id` | `int` or `str` | **Yes** | Unique primary key of the invoice in the ERP. Used for tracking and subset output; strictly excluded from ML training features. | `1001` |
| `customer_id` | `int` | **Yes** | Identifier of the customer. Used to filter out foreign customer invoices before solving. | `41` |
| `value` | `float` | **Yes** | Nominal face value in currency. Converted to exact integer cents (`int(round(value * 100))`) for the Subset Sum solver. | `2000.00` |
| `due_date` | `str` (ISO) | **Yes** | Due date (`YYYY-MM-DD`). Used to compute `days_overdue` and overdue delinquency ratios for ML features. | `"2026-01-10"` |
| `issue_date` | `str` (ISO) | **Yes** | Issuance date (`YYYY-MM-DD`). Used to compute invoice age and seniority metrics for ML features. | `"2025-12-01"` |
| `status` | `str` | Optional | Invoice status (e.g., `"OPEN"`, `"PAID"`, `"CANCELLED"`). Only `"OPEN"` invoices participate in reconciliation. | `"OPEN"` |

#### JSON Format Example:
```json
[
  {
    "id": 1001,
    "customer_id": 41,
    "value": 2000.00,
    "due_date": "2026-01-10",
    "issue_date": "2025-12-01",
    "status": "OPEN"
  },
  {
    "id": 1002,
    "customer_id": 41,
    "value": 3500.00,
    "due_date": "2026-01-20",
    "issue_date": "2025-12-10",
    "status": "OPEN"
  }
]
```

#### CSV Format Example (`accounts_receivable.csv`):
```csv
id,customer_id,value,due_date,issue_date,status
2001,1025,3200.00,2026-01-15,2025-12-01,OPEN
2002,1025,1800.00,2026-01-20,2025-12-10,OPEN
2003,1025,5000.00,2026-04-10,2026-02-01,OPEN
```

> **Automatic Column Aliases**: `UniversalDataLoader.load_invoices_from_csv()` automatically maps common ERP column names:
> - `invoice_id` $\rightarrow$ `id`
> - `gross_amount` / `invoice_amount` / `amount` $\rightarrow$ `value`
> - `client_id` $\rightarrow$ `customer_id`
> - `dt_vencimento` $\rightarrow$ `due_date`
> - `dt_emissao` $\rightarrow$ `issue_date`

---

## 2. Recommended Relational Database Schema

### 2.1 Invoices Table (`erp_invoices`)
```sql
CREATE TABLE erp_invoices (
    id BIGINT PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    invoice_number VARCHAR(64) NOT NULL,
    issue_date DATE NOT NULL,
    due_date DATE NOT NULL,
    gross_amount DECIMAL(15, 2) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN', -- 'OPEN', 'PAID', 'CANCELLED'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_invoices_cust_status ON erp_invoices(customer_id, status);
```

### 2.2 Historical Payment Settlements Table (`erp_payment_events` & `erp_payment_items`)
```sql
CREATE TABLE erp_payment_events (
    payment_id BIGINT PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    payment_date DATE NOT NULL,
    amount_received DECIMAL(15, 2) NOT NULL,
    bank_statement_id VARCHAR(64)
);

CREATE TABLE erp_payment_items (
    payment_id BIGINT REFERENCES erp_payment_events(payment_id),
    invoice_id BIGINT REFERENCES erp_invoices(id),
    cleared_amount DECIMAL(15, 2) NOT NULL,
    PRIMARY KEY (payment_id, invoice_id)
);
```

---

## 3. SQL View for Training Pipeline Ingestion (`vw_training_payment_history`)

```sql
CREATE OR REPLACE VIEW vw_training_payment_history AS
SELECT 
    p.payment_id,
    p.customer_id,
    p.amount_received AS amount,
    p.payment_date,
    ARRAY_AGG(pi.invoice_id) AS selected_invoice_ids
FROM erp_payment_events p
JOIN erp_payment_items pi ON p.payment_id = pi.payment_id
GROUP BY p.payment_id, p.customer_id, p.amount_received, p.payment_date;
```

#### Field Specifications of `vw_training_payment_history`:

| Column Name | Data Type | Required? | Description | Example |
| :--- | :--- | :---: | :--- | :--- |
| `payment_id` | `BIGINT` | **Yes** | Unique identifier of the past settlement event. | `501` |
| `customer_id` | `BIGINT` | **Yes** | Customer who performed the payment. | `41` |
| `amount` | `DECIMAL(15,2)` | **Yes** | Total monetary amount received in that settlement. | `10000.00` |
| `payment_date` | `DATE` / `VARCHAR` | **Yes** | Date when the payment occurred (`YYYY-MM-DD`). | `'2026-02-15'` |
| `selected_invoice_ids`| `ARRAY` / `VARCHAR`| **Yes** | Array or comma-separated list of invoice IDs that were settled. | `ARRAY[1004, 1005]` or `"{1004,1005}"` |

---

## 4. Python Ingestion and Inference Code

```python
from src.data.loaders import UniversalDataLoader
from src.pipeline.inference import suggest_invoice_payments

# Ingest open invoices from CSV or JSON
open_invoices = UniversalDataLoader.load_invoices_from_csv("data/sample/accounts_receivable.csv")
# or: open_invoices = UniversalDataLoader.load_invoices_from_json("data/sample/my_invoices.json")

# Ingest historical training events from SQL database
DB_URI = "postgresql://user:password@localhost:5432/erp_database"
real_history = UniversalDataLoader.load_payment_history_from_sql(
    db_connection_uri=DB_URI,
    events_table_or_query="vw_training_payment_history",
    invoices_table="erp_invoices"
)

# Run end-to-end two-stage reconciliation
result = suggest_invoice_payments(
    customer_id=1025,
    amount=5000.00,
    invoices=open_invoices,
    payment_history=real_history
)

print(result.to_dict())
```

---

## 5. Periodic Retraining (MLOps)

It is recommended to schedule weekly or monthly automated retraining jobs:
1. Query the last 6–12 months of ERP settlement events via `vw_training_payment_history`;
2. Query candidate invoices open at each settlement date via `erp_invoices`;
3. Execute `python3 _mlp/generate_features.py`;
4. Execute `python3 _mlp/train_mlp.py`;
5. Persist weights to `saved_models/pytorch_mlp_weights.pt` and `saved_models/pytorch_scaler.joblib`.
