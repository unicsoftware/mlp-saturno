"""
Deep Learning Neural Network (PyTorch MLP) for ranking candidate invoice payment combinations.

Author: Elpidio Junior E-ABC
License: MIT
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class InvoiceRankingPyTorchNet(nn.Module):
    """
    MLP Architecture in PyTorch:
    Input -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear -> Output
    """

    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout_rate: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepLearningRankingModel:
    """
    Training, evaluation and inference wrapper for the PyTorch MLP Ranking Network.
    """

    def __init__(
        self,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        dropout_rate: float = 0.2,
        lr: float = 0.005,
        epochs: int = 60,
        batch_size: int = 32,
        random_seed: int = 42
    ):
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.scaler = StandardScaler()
        self.model: Optional[InvoiceRankingPyTorchNet] = None
        self.input_dim: int = 0
        self.train_losses: List[float] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, verbose: bool = False) -> "DeepLearningRankingModel":
        self.input_dim = X_train.shape[1]
        X_scaled = self.scaler.fit_transform(X_train)

        self.model = InvoiceRankingPyTorchNet(
            input_dim=self.input_dim,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        num_pos = max(1.0, float(np.sum(y_train == 1)))
        num_neg = float(len(y_train) - num_pos)
        pos_weight = torch.tensor([num_neg / num_pos], device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        self.train_losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)

            epoch_loss /= len(loader.dataset)
            self.train_losses.append(epoch_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Deep Learning model has not been trained yet.")

        self.model.eval()
        X_scaled = self.scaler.transform(X)
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_x)
            probabilities = np.array(torch.sigmoid(logits).cpu().flatten().tolist(), dtype=float)

        return probabilities

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray, metadata_test: list = None) -> Dict[str, float]:
        proba = self.predict_proba(X_test)
        y_pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        top1_acc = self._compute_topk(proba, y_test, metadata_test, k=1) if metadata_test else acc
        top3_acc = self._compute_topk(proba, y_test, metadata_test, k=3) if metadata_test else acc
        top5_acc = self._compute_topk(proba, y_test, metadata_test, k=5) if metadata_test else acc

        return {
            "Model": "PyTorch MLP",
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "Top-1 Accuracy": round(top1_acc, 4),
            "Top-3 Accuracy": round(top3_acc, 4),
            "Top-5 Accuracy": round(top5_acc, 4),
        }

    def _compute_topk(self, y_proba: np.ndarray, y_test: np.ndarray, metadata: list, k: int = 1) -> float:
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
            if not any(item[1] == 1 for item in items):
                continue
            total_groups += 1

            sorted_items = sorted(items, key=lambda x: x[0], reverse=True)
            top_k_items = sorted_items[:k]
            if any(item[1] == 1 for item in top_k_items):
                correct += 1

        return correct / total_groups if total_groups > 0 else 0.0
