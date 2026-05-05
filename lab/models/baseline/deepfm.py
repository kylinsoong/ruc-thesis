import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


class DeepFMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 16,
        hidden_dims: list = [64, 32],
        dropout_rate: float = 0.2
    ):
        super(DeepFMModel, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.linear = nn.Linear(input_dim, 1)

        self.fm_embedding = nn.Linear(input_dim, embed_dim, bias=False)

        deep_input_dim = input_dim + embed_dim
        layers = []
        prev_dim = deep_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.deep = nn.Sequential(*layers)

        self.output = nn.Linear(1 + 1 + hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(x)

        embed = self.fm_embedding(x)
        sum_of_square = torch.sum(embed, dim=1, keepdim=True) ** 2
        square_of_sum = torch.sum(embed ** 2, dim=1, keepdim=True)
        fm_out = 0.5 * (sum_of_square - square_of_sum)

        deep_input = torch.cat([x, embed], dim=1)
        deep_out = self.deep(deep_input)

        combined = torch.cat([linear_out, fm_out, deep_out], dim=1)
        output = torch.sigmoid(self.output(combined))
        return output.squeeze(-1)


class TabularDeepFM:
    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dims: list = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        random_state: int = 42
    ):
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = None
        self.input_dim = None
        self.is_fitted = False
        self.device = torch.device("cpu")

    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> None:
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values.astype(np.float32)
        if isinstance(y_train, pd.Series):
            y_train = y_train.values.astype(np.float32)

        self.input_dim = X_train.shape[1]

        if X_val is None:
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
            )
        else:
            X_train_split = X_train
            y_train_split = y_train
            if isinstance(X_val, pd.DataFrame):
                X_val_split = X_val.values.astype(np.float32)
            if isinstance(y_val, pd.Series):
                y_val_split = y_val.values.astype(np.float32)

        X_train_tensor = torch.from_numpy(X_train_split)
        y_train_tensor = torch.from_numpy(y_train_split)
        X_val_tensor = torch.from_numpy(X_val_split)
        y_val_tensor = torch.from_numpy(y_val_split)

        self.model = DeepFMModel(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            indices = np.random.permutation(len(X_train_tensor))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_x = X_train_tensor[batch_indices]
                batch_y = y_train_tensor[batch_indices]

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        self.is_fitted = True

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not trained yet")

        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)

        X_tensor = torch.from_numpy(X)
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            predictions = (output > 0.5).float()
        return predictions.numpy()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not trained yet")

        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)

        X_tensor = torch.from_numpy(X)
        self.model.eval()
        with torch.no_grad():
            probabilities = self.model(X_tensor)
        return probabilities.numpy()

    def evaluate(self, X_test: Union[np.ndarray, pd.DataFrame], y_test: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        try:
            metrics["auc_roc"] = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            metrics["auc_roc"] = 0.0

        try:
            metrics["auc_pr"] = float(average_precision_score(y_test, y_prob))
        except ValueError:
            metrics["auc_pr"] = 0.0

        return metrics
