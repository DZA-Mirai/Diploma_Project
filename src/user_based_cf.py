# src/user_based_cf.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


@dataclass
class UserBasedCF:
    """
    User-based collaborative filtering with cosine similarity on
    user-centered ratings.

    Workflow:
      model = UserBasedCF(k=20)
      model.fit(train_df)
      rmse, mae = evaluate_df(model, eval_df)
    """
    k: int = 20
    metric: str = "cosine"
    algorithm: str = "brute"
    n_jobs: int = -1

    # learned state
    user_id_map: Optional[Dict[int, int]] = None
    movie_id_map: Optional[Dict[int, int]] = None
    user_means: Optional[np.ndarray] = None
    global_mean: Optional[float] = None
    user_item_matrix: Optional[csr_matrix] = None
    neighbors: Optional[np.ndarray] = None
    distances: Optional[np.ndarray] = None
    knn: Optional[NearestNeighbors] = None

    def fit(self, train_df: pd.DataFrame) -> "UserBasedCF":
        """
        Fit the model:
        - compute user means and centered ratings
        - build CSR matrix (users x items)
        - compute nearest neighbors for each user
        """
        required = {"userId", "movieId", "rating"}
        missing = required - set(train_df.columns)
        if missing:
            raise ValueError(f"train_df is missing columns: {missing}")

        # global mean (fallback)
        self.global_mean = float(train_df["rating"].mean())

        # user means (for de-centering predictions + fallback)
        user_mean_series = train_df.groupby("userId")["rating"].mean()
        # build user map from training set
        user_ids = train_df["userId"].unique()
        self.user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
        n_users = len(self.user_id_map)

        # store user_means as a vector aligned with user_id_map
        self.user_means = np.empty(n_users, dtype=np.float32)
        for uid, idx in self.user_id_map.items():
            self.user_means[idx] = float(user_mean_series.loc[uid])

        # item map from training set
        movie_ids = train_df["movieId"].unique()
        self.movie_id_map = {mid: idx for idx, mid in enumerate(movie_ids)}
        n_movies = len(self.movie_id_map)

        # centered ratings: r_ui - mean_u
        # vectorized mapping
        u_idx = train_df["userId"].map(self.user_id_map).to_numpy(dtype=np.int32)
        i_idx = train_df["movieId"].map(self.movie_id_map).to_numpy(dtype=np.int32)
        # (train_df['userId'].map(user_mean_series) is also fine but slower)
        centered = (train_df["rating"].to_numpy(dtype=np.float32) - self.user_means[u_idx])

        # CSR matrix
        self.user_item_matrix = csr_matrix(
            (centered, (u_idx, i_idx)),
            shape=(n_users, n_movies),
            dtype=np.float32,
        )

        # KNN over users
        self.knn = NearestNeighbors(
            n_neighbors=self.k + 1,  # +1 to include self
            metric=self.metric,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
        )
        self.knn.fit(self.user_item_matrix)

        # precompute neighbors/distances for all users (fast at predict time)
        self.distances, self.neighbors = self.knn.kneighbors(self.user_item_matrix)

        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a single (user_id, movie_id).
        """
        if self.user_id_map is None or self.movie_id_map is None:
            raise RuntimeError("Model is not fitted. Call fit(train_df) first.")
        assert self.user_means is not None
        assert self.user_item_matrix is not None
        assert self.neighbors is not None
        assert self.distances is not None
        assert self.global_mean is not None

        u = self.user_id_map.get(user_id)
        if u is None:
            return self.global_mean

        i = self.movie_id_map.get(movie_id)
        if i is None:
            # user known, item unknown -> user mean fallback
            return float(self.user_means[u])

        return float(self._predict_idx(u, i))

    def predict_many(self, users: np.ndarray, movies: np.ndarray) -> np.ndarray:
        """
        Predict for arrays of userIds/movieIds (same length).
        """
        if len(users) != len(movies):
            raise ValueError("users and movies must have same length")

        preds = np.empty(len(users), dtype=np.float32)
        for t, (u, m) in enumerate(zip(users, movies)):
            preds[t] = self.predict(int(u), int(m))
        return preds

    def _predict_idx(self, u_idx: int, m_idx: int) -> float:
        """
        Core prediction using precomputed neighbors/distances.
        Operates on internal indices.
        """
        assert self.user_means is not None
        assert self.user_item_matrix is not None
        assert self.neighbors is not None
        assert self.distances is not None

        # neighbors[0] is the user itself -> skip it
        neigh_users = self.neighbors[u_idx, 1:]
        neigh_dists = self.distances[u_idx, 1:]

        # cosine distance -> similarity
        sims = 1.0 - neigh_dists

        # pull neighbor centered ratings for this movie
        # if neighbor hasn't rated the movie, matrix entry is 0
        neigh_centered = self.user_item_matrix[neigh_users, m_idx].toarray().ravel()

        # use only neighbors who have a non-zero centered rating for this item
        mask = neigh_centered != 0
        if not np.any(mask):
            # no neighbor info -> user mean
            return float(self.user_means[u_idx])

        sims = sims[mask]
        neigh_centered = neigh_centered[mask]

        denom = np.sum(np.abs(sims))
        if denom < 1e-12:
            return float(self.user_means[u_idx])

        # weighted sum of centered ratings + add back user mean
        pred_centered = float(np.dot(sims, neigh_centered) / denom)
        pred = float(self.user_means[u_idx] + pred_centered)

        # optional clipping for MovieLens scale
        if pred < 0.5:
            pred = 0.5
        elif pred > 5.0:
            pred = 5.0

        return pred

    def save_knn(self, path_prefix: str) -> None:
        """
        Save neighbors/distances arrays to disk.
        """
        if self.neighbors is None or self.distances is None:
            raise RuntimeError("No KNN data to save. Fit the model first.")
        np.save(f"{path_prefix}_neighbors.npy", self.neighbors)
        np.save(f"{path_prefix}_distances.npy", self.distances)

    def load_knn(self, path_prefix: str) -> None:
        """
        Load neighbors/distances arrays from disk.
        """
        self.neighbors = np.load(f"{path_prefix}_neighbors.npy")
        self.distances = np.load(f"{path_prefix}_distances.npy")
