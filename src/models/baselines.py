"""
Classical Machine Learning Baseline Models (Logistic Regression, Random Forest, Gradient Boosting).

Author: Elpidio Junior E-ABC
License: MIT
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class BaselineModelsManager:
    """
    Manager and benchmarking container for classical ML models for combination ranking.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=random_state, max_iter=1000, class_weight="balanced"))
            ]),
            "Random Forest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state, class_weight="balanced"))
            ]),
            "Gradient Boosting": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=random_state))
            ])
        }
        self.trained_models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}

    def fit_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        meta_test: list = None
    ) -> pd.DataFrame:
        """
        Trains all baseline models and computes classification and Top-K ranking accuracy metrics.
        """
        results = []

        for name, pipe in self.models.items():
            pipe.fit(X_train, y_train)
            self.trained_models[name] = pipe

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            top1_acc = self._compute_topk_accuracy(y_proba, y_test, meta_test, k=1) if meta_test else acc
            top3_acc = self._compute_topk_accuracy(y_proba, y_test, meta_test, k=3) if meta_test else acc

            metrics_dict = {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4),
                "Top-1 Accuracy": round(top1_acc, 4),
                "Top-3 Accuracy": round(top3_acc, 4)
            }
            self.metrics[name] = metrics_dict
            results.append(metrics_dict)

        return pd.DataFrame(results)

    def _compute_topk_accuracy(self, y_proba: np.ndarray, y_test: np.ndarray, metadata: list, k: int = 1) -> float:
        """Computes whether ground truth positive combination is among top-K predicted for each payment event."""
        if not metadata or len(metadata) != len(y_proba):
            return 0.0

        groups: Dict[int, list] = {}
        for idx, meta in enumerate(metadata):
            pid = meta["payment_id"]
            if pid not in groups:
                groups[pid] = []
            groups[pid].append((y_proba[idx], y_test[idx]))

        correct = 0
        total_groups = 0

        for pid, items in groups.items():
            has_positive = any(item[1] == 1 for item in items)
            if not has_positive:
                continue
            total_groups += 1

            sorted_items = sorted(items, key=lambda x: x[0], reverse=True)
            top_k_items = sorted_items[:k]
            if any(item[1] == 1 for item in top_k_items):
                correct += 1

        return correct / total_groups if total_groups > 0 else 0.0

    def predict_score(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' has not been trained.")
        pipe = self.trained_models[model_name]
        return pipe.predict_proba(X)[:, 1]
