from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, num_layers=4):
        super().__init__()

        layers = []
        current_dim = in_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPSurrogate:
    """
    Full surrogate operator:

        S_theta: a -> y_hat

    Workflow:

        a -> a_norm -> N_theta(a_norm) = u_norm_hat
          -> u_hat -> y_hat
    """

    def __init__(
        self,
        model_path,
        scaler_path,
        pca_dataset_path,
        config_path=None,
        device=None,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.pca_dataset_path = Path(pca_dataset_path)
        self.config_path = None if config_path is None else Path(config_path)

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_scalers()
        self._load_pca_data()
        self._load_model()

    def _load_scalers(self):
        scalers = np.load(self.scaler_path)

        self.A_mean = scalers["A_mean"]
        self.A_std = scalers["A_std"]
        self.U_mean = scalers["U_mean"]
        self.U_std = scalers["U_std"]

    def _load_pca_data(self):
        data = np.load(self.pca_dataset_path)

        self.Phi = data["Phi"]
        self.Y_mean = data["Y_mean"]
        self.t = data["t"]
        self.r = int(data["r"])

    def _load_model(self):
        in_dim = self.A_mean.shape[0]
        out_dim = self.U_mean.shape[0]

        hidden_dim = 128
        num_layers = 4

        self.model = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)

        state = torch.load(self.model_path,map_location=self.device,weights_only=True,)
        self.model.load_state_dict(state)
        self.model.eval()

    def normalize_A(self, A):
        return (A - self.A_mean) / self.A_std

    def denormalize_U(self, U_norm):
        return U_norm * self.U_std + self.U_mean

    def predict_U(self, A):
        """
        Predict PCA coefficients.

        Input:
            A: shape (6,) or (n_samples, 6)

        Output:
            U_hat: shape (r,) or (n_samples, r)
        """
        A = np.asarray(A, dtype=np.float64)
        single_input = A.ndim == 1

        if single_input:
            A = A[None, :]

        A_norm = self.normalize_A(A)

        with torch.no_grad():
            x = torch.tensor(A_norm, dtype=torch.float32, device=self.device)
            U_norm_hat = self.model(x).cpu().numpy()

        U_hat = self.denormalize_U(U_norm_hat)

        if single_input:
            return U_hat[0]

        return U_hat

    def reconstruct_Y(self, U):
        """
        Reconstruct output signal from PCA coefficients.

        Input:
            U: shape (r,) or (n_samples, r)

        Output:
            Y_hat: shape (N_t,) or (n_samples, N_t)
        """
        U = np.asarray(U, dtype=np.float64)
        single_input = U.ndim == 1

        if single_input:
            U = U[None, :]

        Y_hat = U @ self.Phi + self.Y_mean

        if single_input:
            return Y_hat[0]

        return Y_hat

    def predict_Y(self, A):
        """
        Full surrogate operator:

            S_theta(A) = Y_hat
        """
        U_hat = self.predict_U(A)
        Y_hat = self.reconstruct_Y(U_hat)
        return Y_hat

    def __call__(self, A):
        return self.predict_Y(A)
