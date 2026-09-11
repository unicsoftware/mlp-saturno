## 1. Feature Dictionary (28 Dimensions)

> [!IMPORTANT]
> **Surrogate IDs are strictly excluded from model feature vectors.**
> Invoice IDs and Customer IDs are arbitrary primary keys. The Machine Learning models and PyTorch Neural Network learn 100% from **monetary face values, proportions relative to received amount (`value / amount`), dispersion, standard deviation, delinquency days, and customer behavioral preferences**.

### 1.1 Candidate Combination Monetary Features (Face Value & Proportions)
1. `comb_total_value`: Total monetary sum of combination (always matches `amount`).
2. `comb_number_of_invoices`: Number of invoices in subset ($|S|$).
3. `comb_avg_invoice_value`: Mean face value of invoices in subset.
4. `comb_min_invoice_value`: Minimum face value in subset.
5. `comb_max_invoice_value`: Maximum face value in subset.
6. `comb_value_spread`: Difference between max and min face value (`comb_max_invoice_value - comb_min_invoice_value`).
7. `comb_max_invoice_ratio`: Percentage of total received amount represented by the largest invoice ($\frac{\text{max\_val}}{\text{amount}}$).
8. `comb_min_invoice_ratio`: Percentage of total received amount represented by the smallest invoice ($\frac{\text{min\_val}}{\text{amount}}$).
9. `comb_avg_invoice_ratio`: Mean invoice value ratio to total amount ($\frac{\text{avg\_val}}{\text{amount}}$).
10. `comb_value_std`: Standard deviation of the invoice face values in the combination.

### 1.2 Delinquency and Seniority Features
11. `comb_overdue_count`: Number of overdue invoices in subset ($\text{ref\_date} > \text{due\_date}$).
12. `comb_overdue_ratio`: Proportion of overdue invoices in subset ($\frac{\text{overdue\_count}}{|S|}$).
13. `comb_avg_days_overdue`: Average overdue days for delinquent invoices.
14. `comb_max_days_overdue`: Maximum overdue days in subset.
15. `comb_oldest_age`: Days since issue date of oldest invoice in subset.
16. `comb_newest_age`: Days since issue date of newest invoice in subset.
17. `comb_age_spread`: Age dispersion (`comb_oldest_age - comb_newest_age`).
18. `comb_avg_age`: Mean age of invoices in subset.

### 1.3 Customer Historical Behavioral Features
19. `cust_num_payments`: Total count of recorded past payment events.
20. `cust_avg_payment_amount`: Mean amount of historical payments.
21. `cust_avg_invoices_per_payment`: Mean count of invoices cleared per payment.
22. `cust_avg_days_overdue_hist`: Historical average overdue days of invoices cleared by customer.
23. `cust_pattern_overdue_pref`: Historical preference score for clearing overdue invoices $[0.0, 1.0]$.
24. `cust_pattern_highest_val_pref`: Historical preference score for clearing large face values $[0.0, 1.0]$.
25. `cust_pattern_oldest_pref`: Historical preference score for clearing oldest invoices $[0.0, 1.0]$.

### 1.4 Interaction / Alignment Features
26. `interact_overdue_alignment`: Alignment between combination overdue ratio and customer overdue preference:
    $$\text{overdue\_ratio} \times (1.0 + 2.0 \times \text{cust\_pattern\_overdue\_pref})$$
27. `interact_value_alignment`: Alignment between largest invoice proportion and customer face-value preference:
    $$\left(\frac{\text{comb\_max\_invoice\_value}}{\text{amount} + \epsilon}\right) \times (1.0 + 2.0 \times \text{cust\_pattern\_highest\_val\_pref})$$
28. `interact_age_alignment`: Alignment between mean age and customer seniority preference:
    $$\left(\frac{\text{comb\_avg\_age}}{100}\right) \times (1.0 + 2.0 \times \text{cust\_pattern\_oldest\_pref})$$

---

## 2. Classical Machine Learning Baseline Models

1. **Logistic Regression**: Linear baseline with $L_2$ regularization and balanced class weights.
2. **Random Forest Classifier**: Ensemble of 100 decision trees with depth regularization (`max_depth=6`).
3. **Gradient Boosting Classifier**: Sequential gradient-boosted trees minimizing classification loss.

---

## 3. Deep Learning Model (PyTorch MLP)

### 3.1 Network Architecture
```text
                  Input Layer (24 Features)
                             │
                             ▼
                    Linear (24 -> 64)
                             │
                             ▼
                           ReLU
                             │
                             ▼
                    Dropout (p = 0.20)
                             │
                             ▼
                    Linear (64 -> 32)
                             │
                             ▼
                           ReLU
                             │
                             ▼
                     Linear (32 -> 1)
                             │
                             ▼
                     Sigmoid Activation
                             │
                             ▼
                  Probability Score [0, 1]
```

### 3.2 Optimization & Loss Function
- **Loss Function**: `nn.BCEWithLogitsLoss(pos_weight=pos_weight)` dynamically weighted for class imbalance:
  $$\text{pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$
- **Optimizer**: `optim.Adam(lr=0.005, weight_decay=1e-4)`.
- **Regularization**: Dropout rate of $0.20$ to prevent overfitting on smaller customer histories.

---

## 4. Key Business Metric: Top-1 Accuracy

In real-world bank reconciliation workflows, the primary benchmark is **Top-1 Ranking Accuracy**:

$$\text{Top-1 Accuracy} = \frac{\sum_{p \in \text{Payments}} \mathbb{I}(\text{argmax}_{c \in C_p} \text{Score}(c) = c^*)}{|\text{Payments}|}$$

where $C_p$ is the set of all mathematically valid combinations for payment $p$, and $c^*$ is the combination actually chosen by the user/operator.
