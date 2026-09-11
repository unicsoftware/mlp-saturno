# Test Scenarios Specification Matrix

**Author**: Elpidio Junior E-ABC  
**License**: MIT

This document details the 11 mandatory test scenarios implemented in the test suite (`tests/test_scenarios.py`) and validated interactively in section 17 of the Jupyter Notebook.

---

## Test Scenarios & Results Matrix

| # | Scenario | Input Condition | Expected Behavior | Test Status |
| :---: | :--- | :--- | :--- | :---: |
| **01** | **Single Exact Match** | 3 invoices (2000, 3000, 4500) and amount = 5000 | Returns single exact combination `[101, 102]` with `remaining = 0.00` | **PASSED (OK)** |
| **02** | **Multiple Exact Matches** | 6 invoices with multiple valid subsets for amount = 10000 | Returns all valid candidate subsets ranked by score | **PASSED (OK)** |
| **03** | **No Exact Match** | Open invoices summing to 9000 and amount = 10000 | Returns `NO_EXACT_MATCH` and empty list (never approximates) | **PASSED (OK)** |
| **04** | **Cross-Customer Invoices Ignored** | Invoices containing `customer_id=99` mixed with `customer_id=41` | Foreign customer invoices discarded before solving | **PASSED (OK)** |
| **05** | **Invoices > Amount** | 10000 invoice for 3000 received amount | Invoices exceeding amount pruned immediately in pre-filter | **PASSED (OK)** |
| **06** | **Identical Face Value Invoices** | 3 invoices of 2000 each for 4000 amount | Returns 3 distinct combinations indexed by primary invoice IDs | **PASSED (OK)** |
| **07** | **Amount = Single Invoice** | Single invoice 102 with 7500 face value and amount = 7500 | Returns unitary combination `[102]` with `number_of_invoices = 1` | **PASSED (OK)** |
| **08** | **Amount > Total Debt** | Total debt = 5000 and amount = 50000 | Returns `NO_EXACT_MATCH` immediately | **PASSED (OK)** |
| **09** | **Hundreds of Invoices** | Dataset containing 200 open invoices | Branch and Bound solver executes in sub-second without memory spike | **PASSED (OK)** |
| **10** | **Customer with Sufficient History** | Customer profile prioritizing overdue invoices | PyTorch MLP ranks overdue combination at Top-1 | **PASSED (OK)** |
| **11** | **Cold Start (No History)** | New customer without recorded history | Deterministic Fallback Ranker activated with explicit explanation | **PASSED (OK)** |

---

## Notable Architecture Test Cases

### Test 04: Multitenant Customer Isolation
Guarantees that foreign customer invoices can never be settled in cross-account batches:
```python
invoices = [
    {"id": 101, "customer_id": 41, "value": 2000.00, "status": "OPEN"},
    {"id": 102, "customer_id": 99, "value": 3000.00, "status": "OPEN"},  # Ignored
    {"id": 103, "customer_id": 41, "value": 3000.00, "status": "OPEN"},
]
# Result: Only invoices 101 and 103 participate for customer_id = 41.
```

### Test 06: Identical Values with Unique Primary Keys
In enterprise recurring billing (e.g. SaaS subscriptions), customers frequently hold multiple open invoices with identical face amounts:
- The system indexes and tracks combinations by immutable primary key `id`, ensuring unambiguous identification of titles.

### Test 11: Transparent and Auditable Fallback
When a new customer makes their first platform payment:
- The response returns explicit explainability reasons:
  `"Ranked via standard accounting business rules (fallback for cold-start customer)."`
- Assures full auditability for compliance and accounting teams.
