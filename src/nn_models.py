from src.config import SEED

import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn.init as init
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin


class TabularDataset(Dataset):
    def __init__(self, X_df, y, dtype=torch.float32):
        self.X = torch.tensor(X_df.to_numpy(), dtype=dtype)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.dtype = dtype

    def __len__(self):
        return self.X.shape[0]

    def num_feats(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y


def get_activation(name, trial=None):
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        slope = trial.suggest_float("neg_slope", 1e-3, 1e1, log=True) if trial else 0.01
        return nn.LeakyReLU(negative_slope=slope)
    else:
        raise ValueError(f"Unknown activation {name}")


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size_list,
        in_dim,
        dropouts,
        activation_list,
        weight_init_scheme="xavier_uniform",
        bias_init=0.0,
    ):

        super().__init__()
        if len(dropouts) != len(hidden_size_list):
            raise ValueError(
                f"Expected dropouts to have same length as hidden states ({len(hidden_size_list)}), got {len(dropouts)} instead"
            )

        layers = []
        prev = in_dim
        ########## Build Skeleton #############
        for h, p, act in zip(hidden_size_list, dropouts, activation_list):
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            if p and p > 0:
                layers.append(nn.Dropout(p))
            prev = h
        # Output node
        layers.append(nn.Linear(prev, 1))
        # Build backbone
        self.net = nn.Sequential(*layers)
        self._init_weights(init_scheme=weight_init_scheme, bias_init=bias_init)

    def _init_weights(
        self, init_scheme: str = "xavier_uniform", bias_init: float = 0.0
    ):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                ### Weight initialization ###
                if init_scheme == "xavier_uniform":
                    # Good default when using ReLU/ELU/Tanh with fan_avg scaling
                    init.xavier_uniform_(m.weight)
                elif init_scheme == "kaiming_uniform":
                    # Good with ReLU-like activations (uses fan_in scaling)
                    init.kaiming_uniform_(m.weight, nonlinearity="relu")
                ### Bias initialization ###
                nn.init.constant_(m.bias, bias_init)

    def forward(self, x):
        return self.net(x).squeeze(1)


class TorchNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_size_list,
        dropouts,
        activation_name,
        lr=1e-3,
        weight_decay=0,
        optimizer_str="adam",
        epochs=30,
        batch_size=64,
        weight_init_scheme="xavier_uniform",
        bias_init=0.0,
        device="cpu",
        verbose=0,
        seed=SEED,
    ):
        self.hidden_size_list = hidden_size_list
        self.dropouts = dropouts
        self.activation_name = activation_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_str = optimizer_str
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_init_scheme = weight_init_scheme
        self.bias_init = bias_init
        self.device = device
        self.verbose = verbose
        self.seed = seed
        self.model_ = None
        self._fit_X = None
        self._fit_y = None

    def fit(self, X, y):
        self._fit_X = X.copy()
        self._fit_y = y.copy()
        self.feature_names_in_ = np.array(X.columns)
        in_dim = X.shape[1]

        # Instantiate new activation for each layer
        acts = [get_activation(self.activation_name) for _ in self.hidden_size_list]

        # Build model and send to device
        model = MLP(
            self.hidden_size_list,
            in_dim,
            self.dropouts,
            acts,
            self.weight_init_scheme,
            self.bias_init,
        ).to(self.device)

        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        # Dataset
        ds = TabularDataset(X, y)

        # >>> Seeded DataLoader shuffle for reproducibility <<<
        g = torch.Generator()
        g.manual_seed(self.seed)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, generator=g)

        # Train loop
        model.train()
        for _ in range(self.epochs):
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        self.model_ = model
        self.classes_ = np.unique(y)  # Needed for sklearn ClassifierMixin
        return self

    def predict_proba(self, X):
        self.model_.eval()  # type: ignore
        with torch.no_grad():
            if isinstance(X, pd.DataFrame):
                X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model_(X_tensor)  # type: ignore
            proba = torch.sigmoid(logits)
            return np.column_stack((1 - proba.cpu().numpy(), proba.cpu().numpy()))

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        y_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_proba)


def load_nn_clf(data_path, in_dim, device):
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    h_params = data["h_params"]
    state_dict = data["state_dict"]
    feature_names_in_ = data["feature_names_in_"]
    hidden_size_list = [h_params["hl_1"], h_params["hl_2"]]
    dropouts = [h_params["dr_1"], h_params["dr_2"]]
    if "hl_3" in h_params and "dr_3" in h_params:
        hidden_size_list.append(h_params["hl_3"])
        dropouts.append(h_params["dr_3"])
    activation_name = h_params["act_func_str"]
    num_epochs = h_params["num_epochs"]
    lr = h_params["lr"]
    weight_decay = h_params["weight_decay"]
    batch_size = h_params["batch_size"]

    clf = TorchNNClassifier(
        hidden_size_list=hidden_size_list,
        dropouts=dropouts,
        activation_name=activation_name,
        epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
    )

    acts = [get_activation(clf.activation_name) for _ in clf.hidden_size_list]
    clf.model_ = MLP(
        hidden_size_list=clf.hidden_size_list,
        in_dim=in_dim,
        dropouts=clf.dropouts,
        activation_list=acts,
        weight_init_scheme=clf.weight_init_scheme,
        bias_init=clf.bias_init,
    ).to(device)

    clf.model_.load_state_dict(state_dict)
    clf.model_.to(device)
    clf.model_.eval()
    clf.classes_ = np.array([0, 1])
    clf.feature_names_in_ = feature_names_in_
    return clf
