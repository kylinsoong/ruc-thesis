import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Union, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


class TabNetModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 3,
        gamma: float = 1.3,
        dropout_rate: float = 0.2
    ):
        super(TabNetModel, self).__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        self.bn = nn.BatchNorm1d(input_dim)

        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, n_d + n_a),
            nn.ReLU(),
            nn.BatchNorm1d(n_d + n_a)
        )

        self.attention_layers = nn.ModuleList()
        self.feature_layers = nn.ModuleList()

        for _ in range(n_steps):
            self.attention_layers.append(nn.Sequential(
                nn.Linear(n_a, input_dim),
                nn.Sigmoid()
            ))
            self.feature_layers.append(nn.Sequential(
                nn.Linear(input_dim, n_d + n_a),
                nn.ReLU(),
                nn.BatchNorm1d(n_d + n_a)
            ))

        self.final_mapping = nn.Linear(n_d, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)

        init_transform = self.feature_transform(x)
        d_out = init_transform[:, :self.n_d]
        a_out = init_transform[:, self.n_d:]

        for step in range(self.n_steps):
            mask = self.attention_layers[step](a_out)
            masked_x = x * mask

            step_transform = self.feature_layers[step](masked_x)
            d_step = step_transform[:, :self.n_d]
            a_step = step_transform[:, self.n_d:]

            d_out = d_out + self.gamma * d_step
            a_out = a_out + a_step

        d_out = self.dropout(d_out)
        output = torch.sigmoid(self.final_mapping(d_out))
        return output.squeeze()


class SimpleTabNet:
    def __init__(
        self,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 3,
        gamma: float = 1.3,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.02,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 15,
        random_state: int = 42
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model = None
        self.input_dim = None
        self.is_fitted = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.model = TabNetModel(
            input_dim=self.input_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        X_train_tensor = torch.FloatTensor(X_train_split).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_split).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_split).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_split).to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.max_epochs):
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
                if patience_counter >= self.patience:
                    break

        self.is_fitted = True

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not trained yet")

        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)

        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            predictions = (output > 0.5).float()
        return predictions.cpu().numpy()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not trained yet")

        if isinstance(X, pd.DataFrame):
            X = X.values.astype(np.float32)

        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            probabilities = self.model(X_tensor)
        return probabilities.cpu().numpy()

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
