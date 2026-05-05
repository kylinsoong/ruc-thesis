"""
深度学习模型模块

实现以下模型：
- DeepFM: 结合FM和DNN的推荐模型
- TabNet: 基于注意力机制的表格数据模型
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Tuple, List, Union
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FMComponent(nn.Module):
    """
    因子分解机(Factorization Machine)组件
    
    实现二阶特征交互：
    y_fm = <w, x> + sum_{i<j}<v_i, v_j> x_i x_j
    """
    
    def __init__(self, num_features: int, embedding_dim: int = 10):
        super(FMComponent, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        self.linear = nn.Linear(num_features, 1)
        self.embedding = nn.Embedding(num_features, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_part = self.linear(x)
        
        feature_indices = torch.arange(self.num_features, device=x.device)
        embeddings = self.embedding(feature_indices)
        
        square_of_sum = torch.pow(torch.matmul(x, embeddings), 2).sum(dim=1, keepdim=True)
        sum_of_square = torch.matmul(torch.pow(x, 2), torch.pow(embeddings, 2)).sum(dim=1, keepdim=True)
        
        fm_part = 0.5 * (square_of_sum - sum_of_square)
        
        return linear_part + fm_part


class DeepComponent(nn.Module):
    """
    深度神经网络组件
    
    多层感知机，用于学习高阶特征交互
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super(DeepComponent, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.deep = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deep(x)


class DeepFM(nn.Module):
    """
    DeepFM模型
    
    结合FM和DNN的优势：
    - FM组件：学习二阶特征交互
    - Deep组件：学习高阶特征交互
    
    支持稀疏特征和稠密特征
    """
    
    def __init__(
        self,
        num_features: int,
        embedding_dim: int = 10,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super(DeepFM, self).__init__()
        
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        self.fm = FMComponent(num_features, embedding_dim)
        self.deep = DeepComponent(
            input_dim=num_features,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )
        
        self.final_layer = nn.Linear(1 + self.deep.output_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fm_output = self.fm(x)
        deep_output = self.deep(x)
        
        combined = torch.cat([fm_output, deep_output], dim=1)
        output = self.final_layer(combined)
        
        return output.squeeze(-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return probs


class DeepFMModel:
    """
    DeepFM模型封装类
    
    提供sklearn风格的接口，包含训练和预测方法
    """
    
    def __init__(
        self,
        embedding_dim: int = 10,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 10,
        early_stopping_patience: int = 5,
        use_batch_norm: bool = True,
        random_state: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_batch_norm = use_batch_norm
        self.random_state = random_state
        
        self.model = None
        self.num_features = None
        self.is_fitted = False
        
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def _prepare_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Tuple[DataLoader, Optional[torch.Tensor]]:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None and isinstance(y, pd.Series):
            y = y.values
        
        X_tensor = torch.FloatTensor(X).to(device)
        
        if y is not None:
            y_tensor = torch.FloatTensor(y).to(device)
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            return dataloader, None
        else:
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            return dataloader, X_tensor
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> 'DeepFMModel':
        if isinstance(X, pd.DataFrame):
            self.num_features = X.shape[1]
        else:
            self.num_features = X.shape[1]
        
        self.model = DeepFM(
            num_features=self.num_features,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        train_loader, _ = self._prepare_data(X, y)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader, _ = self._prepare_data(X_val, y_val)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    break
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        dataloader, X_tensor = self._prepare_data(X)
        
        predictions = []
        with torch.no_grad():
            for (batch_x,) in dataloader:
                outputs = self.model(batch_x)
                preds = (torch.sigmoid(outputs) > 0.5).int()
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        dataloader, _ = self._prepare_data(X)
        
        probabilities = []
        with torch.no_grad():
            for (batch_x,) in dataloader:
                outputs = self.model(batch_x)
                probs = torch.sigmoid(outputs)
                probabilities.extend(probs.cpu().numpy())
        
        probs_array = np.array(probabilities)
        return np.column_stack([1 - probs_array, probs_array])
    
    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_features': self.num_features,
            'embedding_dim': self.embedding_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'DeepFMModel':
        checkpoint = torch.load(filepath, map_location=device)
        
        self.num_features = checkpoint['num_features']
        self.embedding_dim = checkpoint['embedding_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.dropout_rate = checkpoint['dropout_rate']
        self.use_batch_norm = checkpoint['use_batch_norm']
        
        self.model = DeepFM(
            num_features=self.num_features,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True
        print(f"Model loaded from {filepath}")
        return self


class TabNetModel:
    """
    TabNet模型封装类
    
    使用pytorch_tabnet库实现，提供sklearn风格接口
    """
    
    def __init__(
        self,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-3,
        momentum: float = 0.3,
        clip_value: float = 2.0,
        optimizer_params: Optional[Dict] = None,
        scheduler_params: Optional[Dict] = None,
        batch_size: int = 1024,
        virtual_batch_size: int = 128,
        num_workers: int = 0,
        max_epochs: int = 100,
        patience: int = 10,
        random_state: int = 42,
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.clip_value = clip_value
        self.optimizer_params = optimizer_params or {'lr': 2e-2}
        self.scheduler_params = scheduler_params
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        
        self.model = None
        self.is_fitted = False
    
    def _create_model(self, input_dim: int, output_dim: int = 1):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            import torch.optim as optim
        except ImportError:
            raise ImportError(
                "pytorch_tabnet未安装。\n"
                "请运行: pip install pytorch-tabnet"
            )
        
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            lambda_sparse=self.lambda_sparse,
            momentum=self.momentum,
            clip_value=self.clip_value,
            optimizer_fn=optim.Adam,
            optimizer_params=self.optimizer_params,
            mask_type='entmax',
            verbose=0,
            seed=self.random_state,
        )
        
        return self.model
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        eval_metric: Optional[List] = None,
    ) -> 'TabNetModel':
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        
        input_dim = X.shape[1]
        self._create_model(input_dim)
        
        if eval_metric is None:
            from pytorch_tabnet.metrics import Metric
            class AUC(Metric):
                def __init__(self):
                    self._name = "auc"
                    self._maximize = True
                
                def __call__(self, y_true, y_pred):
                    from sklearn.metrics import roc_auc_score
                    try:
                        return roc_auc_score(y_true, y_pred[:, 1])
                    except:
                        return 0.5
            eval_metric = [AUC]
        
        X_val_arr = None
        y_val_arr = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val_arr = X_val.values.astype(np.float32)
            else:
                X_val_arr = X_val.astype(np.float32)
            if isinstance(y_val, pd.Series):
                y_val_arr = y_val.values.astype(np.int64)
            else:
                y_val_arr = y_val.astype(np.int64)
        
        self.model.fit(
            X_train=X,
            y_train=y,
            eval_set=[(X_val_arr, y_val_arr)] if X_val_arr is not None else None,
            eval_name=['valid'] if X_val_arr is not None else None,
            eval_metric=eval_metric,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            num_workers=self.num_workers,
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = X.astype(np.float32)
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = X.astype(np.float32)
        return self.model.predict_proba(X)
    
    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'TabNetModel':
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError:
            raise ImportError(
                "pytorch_tabnet未安装。\n"
                "请运行: pip install pytorch-tabnet"
            )
        
        self.model = TabNetClassifier()
        self.model.load_model(filepath)
        self.is_fitted = True
        print(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importances(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.feature_importances_


def create_deep_learning_model(
    model_type: str,
    **kwargs
) -> Union[DeepFMModel, TabNetModel]:
    """
    创建深度学习模型的工厂函数
    
    Args:
        model_type: 模型类型，'deepfm' 或 'tabnet'
        **kwargs: 模型参数
        
    Returns:
        模型实例
    """
    model_type = model_type.lower()
    
    if model_type == 'deepfm':
        return DeepFMModel(**kwargs)
    elif model_type == 'tabnet':
        return TabNetModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'deepfm', 'tabnet'")


if __name__ == "__main__":
    print("Testing DeepFM model...")
    
    np.random.seed(42)
    X = np.random.randn(1000, 20).astype(np.float32)
    y = np.random.randint(0, 2, 1000).astype(np.float32)
    
    model = DeepFMModel(
        embedding_dim=8,
        hidden_dims=[64, 32],
        epochs=5,
        batch_size=64,
    )
    
    model.fit(X[:800], y[:800], X[800:], y[800:])
    
    predictions = model.predict(X[800:])
    probabilities = model.predict_proba(X[800:])
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print("DeepFM test passed!")
