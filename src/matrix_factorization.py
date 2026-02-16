from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import math
import random
import numpy as np
import pandas as pd
import time

Rating = Tuple[Union[int, str], Union[int, str], float]  # (user_id, item_id, rating)


class BiasedMF:
    """
    Biased Matrix Factorization using SGD.

    Parameters
    ----------
    n_factors : int
        Latent dimension k
    lr : float
        Learning rate
    reg : float
        L2 regularization strength
    n_epochs : int
        Number of passes over the training set
    seed : int
        Random seed for reproducibility
    init_std : float
        Std-dev for normal initialization of factors
    """

    def __init__(
            self,
            n_factors: int = 50,
            lr: float = 0.005,
            reg: float = 0.02,
            n_epochs: int = 20,
            seed: int = 42,
            init_std: float = 0.1,
            verbose: bool = True,
    ) -> None:
        if n_factors <= 0:
            raise ValueError("n_factors must be positive.")
        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.reg = float(reg)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        self.init_std = float(init_std)
        self.verbose = bool(verbose)

        # Learned parameters (initialized in fit)
        self.mu: float = 0.0
        self.user_map: Dict[Union[int, str], int] = {}
        self.item_map: Dict[Union[int, str], int] = {}
        self.user_inv: List[Union[int, str]] = []
        self.item_inv: List[Union[int, str]] = []

        self.P: Optional[np.ndarray] = None  # shape: (n_users, k)
        self.Q: Optional[np.ndarray] = None  # shape: (n_items, k)
        self.bu: Optional[np.ndarray] = None  # shape: (n_users,)
        self.bi: Optional[np.ndarray] = None  # shape: (n_items,)

        self._rng = np.random.default_rng(self.seed)

    # ---------------------------
    # ID mapping / initialization
    # ---------------------------

    def _to_index_arrays(self, train_df, eval_df=None):
        # Factorize TRAIN only
        u_codes, u_uniques = pd.factorize(train_df["userId"], sort=False)
        i_codes, i_uniques = pd.factorize(train_df["movieId"], sort=False)

        r = train_df["rating"].to_numpy(dtype=np.float32)
        u = u_codes.astype(np.int32, copy=False)
        i = i_codes.astype(np.int32, copy=False)

        # Store mappings in the model (so predict/recommend works)
        self.user_inv = list(u_uniques)
        self.item_inv = list(i_uniques)
        self.user_map = {uid: idx for idx, uid in enumerate(self.user_inv)}
        self.item_map = {mid: idx for idx, mid in enumerate(self.item_inv)}

        if eval_df is None:
            return (u, i, r), None

        eu = eval_df["userId"].map(self.user_map).to_numpy()
        ei = eval_df["movieId"].map(self.item_map).to_numpy()
        er = eval_df["rating"].to_numpy(dtype=np.float32)

        # Warm-start only: drop unknown users/items
        mask = (~pd.isna(eu)) & (~pd.isna(ei))
        eu = eu[mask].astype(np.int32, copy=False)
        ei = ei[mask].astype(np.int32, copy=False)
        er = er[mask].astype(np.float32, copy=False)

        return (u, i, r), (eu, ei, er)

    def _init_params(self) -> None:
        n_users = len(self.user_map)
        n_items = len(self.item_map)

        self.P = self._rng.normal(0.0, self.init_std, size=(n_users, self.n_factors)).astype(np.float32)
        self.Q = self._rng.normal(0.0, self.init_std, size=(n_items, self.n_factors)).astype(np.float32)
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)

    # Checkpoints
    def save_checkpoint(self, path: str):
        np.savez_compressed(
            path,
            mu=self.mu,
            bu=self.bu,
            bi=self.bi,
            P=self.P,
            Q=self.Q,
            user_inv=np.array(self.user_inv, dtype=object),
            item_inv=np.array(self.item_inv, dtype=object),
            best_epoch=getattr(self, "best_epoch_", None),
            best_val_rmse=getattr(self, "best_val_rmse_", None),
            n_factors=self.n_factors,
            lr=self.lr,
            reg=self.reg,
        )

    def load_checkpoint(self, path: str):
        import numpy as np

        data = np.load(path, allow_pickle=True)

        self.mu = float(data["mu"])
        self.bu = data["bu"]
        self.bi = data["bi"]
        self.P = data["P"]
        self.Q = data["Q"]

        self.user_inv = list(data["user_inv"])
        self.item_inv = list(data["item_inv"])
        self.user_map = {u: i for i, u in enumerate(self.user_inv)}
        self.item_map = {i: j for j, i in enumerate(self.item_inv)}

        self.best_epoch_ = data.get("best_epoch", None)
        self.best_val_rmse_ = data.get("best_val_rmse", None)

    # ---------------------------
    # Core prediction helpers
    # ---------------------------

    def _check_fitted(self) -> None:
        if self.P is None or self.Q is None or self.bu is None or self.bi is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    def _known_user_item(self, user_id: Union[int, str], item_id: Union[int, str]) -> Tuple[
        Optional[int], Optional[int]]:
        u = self.user_map.get(user_id)
        i = self.item_map.get(item_id)
        return u, i

    def predict(self, user_id: Union[int, str], item_id: Union[int, str],
                clip: Tuple[float, float] = (0.5, 5.0)) -> float:
        """
        Predict rating for (user_id, item_id).
        For unknown user or item: fallback to mu + known bias if available.

        clip: clamp prediction to rating scale (MovieLens: 0.5..5.0)
        """
        self._check_fitted()
        u_idx, i_idx = self._known_user_item(user_id, item_id)

        pred = self.mu
        if u_idx is not None:
            pred += float(self.bu[u_idx])
        if i_idx is not None:
            pred += float(self.bi[i_idx])
        if u_idx is not None and i_idx is not None:
            pred += float(np.dot(self.P[u_idx], self.Q[i_idx]))

        lo, hi = clip
        if lo is not None and pred < lo:
            pred = lo
        if hi is not None and pred > hi:
            pred = hi
        return float(pred)

    # ---------------------------
    # Training / evaluation
    # ---------------------------

    def fit(self, train_ratings, val_ratings=None, patience=2) -> "BiasedMF":
        """
        Fit model with SGD.

        train_ratings: list of (user_id, item_id, rating)
        val_ratings: optional validation set
        """
        print(f"Training is starting with parameters {self.n_factors}, {self.lr}, {self.reg}, {self.n_epochs}")
        if len(train_ratings) == 0:
            raise ValueError("train_ratings is empty.")

        # Build maps only from training data
        (u, i, r), val_tuple = self._to_index_arrays(train_ratings, val_ratings)

        # Global mean on training
        self.mu = float(r.mean())

        self._init_params()

        idx = np.arange(len(r), dtype=np.int32)

        best_val_rmse = np.inf
        best_epoch = 0
        epochs_without_improvement = 0

        # To restore best model
        best_state = None

        # SGD
        for epoch in range(1, self.n_epochs + 1):
            print(f"Epoch number {epoch} started:\n")
            start = time.perf_counter()
            self._rng.shuffle(idx)

            for t in idx:
                uu = u[t]
                ii = i[t]
                rr = r[t]

                # prediction
                pred = self.mu + self.bu[uu] + self.bi[ii] + float(np.dot(self.P[uu], self.Q[ii]))
                err = rr - pred

                # IMPORTANT: avoid .copy() allocations
                pu = self.P[uu]
                qi = self.Q[ii]

                # store old pu (as a small copy) ONLY if needed
                pu_old = pu.copy()

                # biases
                self.bu[uu] += self.lr * (err - self.reg * self.bu[uu])
                self.bi[ii] += self.lr * (err - self.reg * self.bi[ii])

                # factors
                pu += self.lr * (err * qi - self.reg * pu)
                qi += self.lr * (err * pu_old - self.reg * qi)

            print(f"Time to finish epoch number {epoch}: {time.perf_counter() - start} s")

            if val_tuple is not None:
                eu, ei, er = val_tuple
                val_rmse, val_mae = self._eval_arrays(eu, ei, er)
                if self.verbose:
                    print(f"Epoch {epoch:02d}/{self.n_epochs} | val RMSE {val_rmse:.5f} MAE {val_mae:.5f} ")

                # Early stopping if no improvement
                if val_rmse < best_val_rmse - 1e-3:
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    epochs_without_improvement = 0

                    # Save best model state
                    best_state = (
                        self.bu.copy(),
                        self.bi.copy(),
                        self.P.copy(),
                        self.Q.copy(),
                    )

                    # Save to disk
                    self.save_checkpoint(r"C:\Users\dza\Desktop\Diploma Thesis\Code\MF\best_model.npz")
                else:
                    epochs_without_improvement += 1

                    if epochs_without_improvement >= patience:
                        if self.verbose:
                            print(
                                f"Early stopping at epoch {epoch}. "
                                f"Best RMSE={best_val_rmse:.5f} at epoch {best_epoch}."
                            )
                        break
            else:
                if self.verbose:
                    print(f"Epoch {epoch:02d}/{self.n_epochs} done")

        return self

    def _eval_arrays(self, u, i, r, clip=(0.5, 5.0)):
        preds = self.mu + self.bu[u] + self.bi[i] + np.sum(self.P[u] * self.Q[i], axis=1)
        lo, hi = clip
        preds = np.clip(preds, lo, hi)

        err = r - preds
        rmse = float(np.sqrt(np.mean(err * err)))
        mae = float(np.mean(np.abs(err)))
        return rmse, mae