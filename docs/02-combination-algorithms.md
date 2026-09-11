# Combination Algorithms - Subset Sum Problem

**Author**: Elpidio Junior E-ABC  
**License**: MIT

---

## 1. Mathematical Formulation

Given a set of $N$ candidate open invoices belonging to a customer:
$$I = \{ (id_1, v_1), (id_2, v_2), \dots, (id_N, v_N) \}$$
where $v_i \in \mathbb{N}^+$ represents the invoice face value in integer cents, and a target received amount $W \in \mathbb{N}^+$ in integer cents, the problem seeks to find all subsets $S \subseteq I$ such that:

$$\sum_{(id_i, v_i) \in S} v_i = W$$

Subject to the following business constraints:
1. $|S| \ge 1$;
2. Each invoice can appear at most once in $S$;
3. Only invoices belonging to the specified customer are allowed;
4. $v_i \le W, \forall (id_i, v_i) \in S$.

---

## 2. Algorithm Comparison Table

| Algorithm | Theoretical Complexity | Memory Usage | Pruning Strategy | Optimal Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Brute Force** | $O(2^N)$ | $O(N)$ | Basic upper cap ($v_i \le W$) | $N \le 12$ (Baseline validation) |
| **Backtracking** | $O(2^N)$ worst, $O(N \log N + k)$ avg | $O(N)$ | Suffix Sums + Descending Sort | $N \le 30$ (Standard production) |
| **Dynamic Programming** | $O(N \cdot W)$ | $O(W \cdot \text{comb})$ | Memory capped state tracking | Small $W$ and moderate $N$ |
| **Branch and Bound** | $O(b^d)$ with $b \ll 2$ | $O(N)$ stack | Dynamic Upper and Lower bounds | $N > 30$ and hundreds of invoices |

---

## 3. Pruning and Optimization Techniques

### 3.1 Suffix Sum Pruning (Backtracking & Branch and Bound)
Let $v_{(1)} \ge v_{(2)} \ge \dots \ge v_{(N)}$ be invoices sorted in descending order of value.
We precompute the suffix sums array:
$$\text{suffix\_sum}[i] = \sum_{j=i}^{N} v_{(j)}$$

During recursive tree traversal at level $i$ with accumulated sum $S_{\text{current}}$:
- **Lower Bound Feasibility Pruning**: If $S_{\text{current}} + \text{suffix\_sum}[i] < W$, it is mathematically impossible to reach target $W$ even if all remaining invoices are chosen. The entire subtree is pruned.
- **Upper Bound Pruning**: If $S_{\text{current}} + v_{(i)} > W$, adding invoice $i$ exceeds the target amount. The branch is pruned immediately.

### 3.2 Memory Safety: `MAX_COMBINATIONS`
In large accounts receivable books with repetitive invoice amounts, the number of valid subsets can grow exponentially. The `MAX_COMBINATIONS` parameter (default: 50–100) bounds memory consumption and prevents runaway CPU time.

---

## 4. Intelligent Factory Strategy (`get_solver`)

`get_solver(method="auto", n_invoices=N)` adapts its choice automatically:
- If $N \le 12$: Uses `BacktrackingSolver` (< 1ms);
- If $12 < N \le 30$: Uses `BacktrackingSolver` with suffix sum pruning (< 5ms);
- If $N > 30$: Uses `BranchAndBoundSolver` bounding depth and dominated nodes (< 20ms).
