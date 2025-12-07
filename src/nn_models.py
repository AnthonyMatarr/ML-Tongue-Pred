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
    """
    PyTorch Dataset wrapper for tabular binary classification data.

    Converts pandas DataFrames and numpy arrays into PyTorch tensors for use
    with DataLoader and neural network training.

    Parameters
    ----------
    X_df : pd.DataFrame
        Feature data.
    y : np.ndarray
        Binary labels (0 or 1).
    dtype : torch.dtype, default=torch.float32
        Data type for feature tensors.
    """

    def __init__(self, X_df, y, dtype=torch.float32):
        self.X = torch.tensor(X_df.to_numpy(), dtype=dtype)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.dtype = dtype

    def __len__(self):
        return self.X.shape[0]

    def num_feats(self):
        """Return number of features in the dataset."""
        return self.X.shape[1]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y


def get_activation(name, trial=None):
    """
    Factory function to create activation layers.

    Parameters
    ----------
    name : str
        Activation function name. Supported: 'relu', 'leaky_relu'.
    trial : optuna.Trial, optional
        Optuna trial object for hyperparameter tuning. If provided and
        name='leaky_relu', suggests negative slope parameter.

    Returns
    -------
    nn.Module
        PyTorch activation layer.

    Raises
    ------
    ValueError
        If activation name is not one of 'relu', 'leaky_relu'.
    """
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky_relu":
        slope = trial.suggest_float("neg_slope", 1e-3, 1e1, log=True) if trial else 0.01
        return nn.LeakyReLU(negative_slope=slope)
    else:
        raise ValueError(f"Unknown activation {name}")


class MLP(nn.Module):
    """
    Multi-layer perceptron for binary classification.

    Configurable feedforward neural network with customizable hidden layers,
    dropout rates, activation functions, and weight initialization schemes.

    Parameters
    ----------
    hidden_size_list : list of int
        Number of units in each hidden layer. Length determines network depth.
    in_dim : int
        Input feature dimensionality.
    dropouts : list of float
        Dropout probability for each hidden layer. Must match length of
        hidden_size_list. Use 0 or None to skip dropout for a layer.
    activation_list : list of nn.Module
        Activation function instances for each hidden layer.
    weight_init_scheme : {'xavier_uniform', 'kaiming_uniform'}, default='xavier_uniform'
        Weight initialization strategy:
        - 'xavier_uniform': Good for tanh/sigmoid/ReLU with fan_avg scaling
        - 'kaiming_uniform': Good for ReLU-like activations with fan_in scaling
    bias_init : float, default=0.0
        Constant value for bias initialization.

    Raises
    ------
    ValueError
        If dropouts length doesn't match hidden_size_list length.

    Notes
    -----
    - Output layer is a single node (binary classification)
    - Uses BCEWithLogitsLoss in training (logits output, not probabilities)
    """

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
        """Initialize weights and biases for all linear layers."""
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
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_dim).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size,) for binary classification.
        """
        return self.net(x).squeeze(1)


class TorchNNClassifier(ClassifierMixin, BaseEstimator):
    """
    Scikit-learn compatible PyTorch neural network classifier.

    Feedforward neural network for binary classification with sklearn interface,
    enabling integration with sklearn pipelines, cross-validation, and calibration.

    Parameters
    ----------
    hidden_size_list : list of int
        Number of units in each hidden layer (e.g., [128, 64]).
    dropouts : list of float
        Dropout probability for each hidden layer. Must match hidden_size_list length.
    activation_name : str
        Activation function name ('relu' or 'leaky_relu').
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    weight_decay : float, default=0
        L2 regularization parameter (weight decay) for Adam.
    optimizer_str : str, default='adam'
        Optimizer name (currently only 'adam' is implemented).
    epochs : int, default=30
        Number of training epochs.
    batch_size : int, default=64
        Mini-batch size for training.
    weight_init_scheme : {'xavier_uniform', 'kaiming_uniform'}, default='xavier_uniform'
        Weight initialization strategy.
    bias_init : float, default=0.0
        Bias initialization constant.
    device : str, default='cpu'
        Device for training ('cpu' or 'cuda').
    verbose : int, default=0
        Verbosity level (currently unused).
    seed : int, default=SEED
        Random seed for DataLoader shuffling (ensures reproducible training).

    Attributes
    ----------
    model_ : MLP
        Trained neural network model (available after fit).
    classes_ : np.ndarray
        Unique class labels (set during fit).
    feature_names_in_ : np.ndarray
        Feature names from training data (set during fit).

    Notes
    -----
    - Uses BCEWithLogitsLoss for training (combines sigmoid and BCE for numerical stability)
    - DataLoader shuffle is seeded for reproducibility
    - Compatible with sklearn's CalibratedClassifierCV for probability calibration
    - Model is set to eval mode during prediction

    Examples
    --------
    >>> from sklearn.calibration import CalibratedClassifierCV
    >>>
    >>> # Basic usage
    >>> clf = TorchNNClassifier(
    ...     hidden_size_list=[128, 64],
    ...     dropouts=[0.3, 0.2],
    ...     activation_name='relu',
    ...     lr=1e-3,
    ...     epochs=50,
    ...     batch_size=64,
    ...     device='cuda'
    ... )
    >>> clf.fit(X_train, y_train)
    >>> y_proba = clf.predict_proba(X_test)
    >>>
    >>> # With calibration
    >>> calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
    >>> calibrated_clf.fit(X_train, y_train)
    >>> y_proba_cal = calibrated_clf.predict_proba(X_test)
    """

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
        """
        Train the neural network on provided data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : np.ndarray
            Binary training labels (0 or 1).

        Returns
        -------
        self
            Fitted classifier instance.
        """
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
        ## Reset
        self._fit_X = None
        self._fit_y = None
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature data for prediction.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2) with probabilities for [class 0, class 1].
        """
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
        """
        Predict binary class labels using 0.5 threshold.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature data for prediction.

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1).
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def score(self, X, y):
        """
        Calculate AUROC score on provided data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature data.
        y : np.ndarray
            True binary labels.

        Returns
        -------
        float
            Area under the ROC curve.
        """

        y_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_proba)


def load_nn_clf(data_path, in_dim, device):
    """
    Load a trained TorchNNClassifier from a saved checkpoint.

    Reconstructs the model architecture from saved hyperparameters and loads
    trained weights from the state dictionary. Used for model deployment and
    inference on new data.

    Parameters
    ----------
    data_path : str or pathlib.Path
        Path to saved model checkpoint (.pt or .pth file) containing:
        - 'h_params': Dictionary of hyperparameters
        - 'state_dict': Model weights
        - 'feature_names_in_': Training feature names
    in_dim : int
        Input feature dimensionality (must match training data).
    device : str
        Device for inference ('cpu' or 'cuda').

    Returns
    -------
    TorchNNClassifier
        Loaded classifier in eval mode, ready for prediction.

    Notes
    -----
    - Model is automatically set to eval mode after loading
    - Supports 2 or 3 hidden layer architectures
    - Feature names are preserved for validation during inference

    Expected checkpoint structure (saved during training):
    ```
    torch.save({
        'h_params': {
            'hl_1': 128, 'hl_2': 64, 'hl_3': 32,  # Hidden layer sizes
            'dr_1': 0.3, 'dr_2': 0.2, 'dr_3': 0.1,  # Dropout rates
            'act_func_str': 'relu',
            'num_epochs': 50,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 64
        },
        'state_dict': model.state_dict(),
        'feature_names_in_': X_train.columns.values
    }, 'model_checkpoint.pt')
    ```
    """
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
