# src/item_based_cf.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


@dataclass
class ItemBasedCF:
    """
    Item-based collaborative filtering with cosine similarity on
    item-centered ratings.

    Workflow:
      model = UserBasedCF(k=20)
      model.fit(train_df)
      rmse, mae = evaluate_df(model, eval_df)
    """
    k: int = 50
    metric: str = "cosine"
    algorithm: str = "brute"
    n_jobs: int = -1

    # learned state
    user_id_map: Optional[Dict[int, int]] = None
    movie_id_map: Optional[Dict[int, int]] = None
    item_means: Optional[np.ndarray] = None
    global_mean: Optional[float] = None
    item_user_matrix: Optional[csr_matrix] = None
    neighbors: Optional[np.ndarray] = None
    distances: Optional[np.ndarray] = None
    knn: Optional[NearestNeighbors] = None

    def fit(self, train_df: pd.DataFrame) -> "ItemBasedCF":
        """
        Fit the model:
        - compute item means and centered ratings
        - build CSR matrix (items x users)
        - compute nearest neighbors for each item
        """
        required = {"userId", "movieId", "rating"}
        missing = required - set(train_df.columns)
        if missing:
            raise ValueError(f"train_df missing columns: {missing}")

        # global mean (fallback)
        self.global_mean = float(train_df["rating"].mean())

        # user and item map from training set
        user_ids = train_df["userId"].unique()
        self.user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
        n_users = len(self.user_id_map)

        movie_ids = train_df["movieId"].unique()
        self.movie_id_map = {mid: idx for idx, mid in enumerate(movie_ids)}
        n_movies = len(self.movie_id_map)

        # item means
        item_mean_series = train_df.groupby("movieId")["rating"].mean()
        self.item_means = np.zeros(len(self.movie_id_map), dtype=np.float32)
        for mid, idx in self.movie_id_map.items():
            self.item_means[idx] = float(item_mean_series.loc[mid])

        # centered ratings: r_ui - mean_i
        # vectorized mapping
        u_idx = train_df["userId"].map(self.user_id_map).to_numpy(dtype=np.int32)
        i_idx = train_df["movieId"].map(self.movie_id_map).to_numpy(dtype=np.int32)
        # (train_df['userId'].map(user_mean_series) is also fine but slower)
        centered = (train_df["rating"].to_numpy(dtype=np.float32) - self.item_means[i_idx])

        # CSR matrix
        self.item_user_matrix = csr_matrix(
            (centered, (i_idx, u_idx)),
            shape=(n_movies, n_users),
            dtype=np.float32
        )

        # KNN over items
        self.knn = NearestNeighbors(
            n_neighbors=self.k + 1,     # +1 to include self
            metric=self.metric,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs
        )
        self.knn.fit(self.item_user_matrix)

        # precompute neighbors/distances for all items (fast at predict time)
        self.distances, self.neighbors = self.knn.kneighbors(self.item_user_matrix)

        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a single (user_id, movie_id).
        """
        if self.user_id_map is None or self.movie_id_map is None:
            raise RuntimeError("Model is not fitted. Call fit(train_df) first.")
        assert self.item_means is not None
        assert self.item_user_matrix is not None
        assert self.neighbors is not None
        assert self.distances is not None
        assert self.global_mean is not None

        u = self.user_id_map.get(user_id)
        if u is None:
            return self.global_mean     # unknown user -> fallback

        i = self.movie_id_map.get(movie_id)
        if i is None:
            return self.global_mean     # unknown item -> fallback

        return float(self._predict_idx(i, u))

    def _predict_idx(self, m_idx: int, u_idx: int) -> float:
        """
        Core prediction using precomputed neighbors/distances.
        Operates on internal indices.
        """
        assert self.item_means is not None
        assert self.item_user_matrix is not None
        assert self.neighbors is not None
        assert self.distances is not None

        # neighbors[0] is the user itself -> skip it
        neigh_items = self.neighbors[m_idx, 1:]
        neigh_dists = self.distances[m_idx, 1:]

        # cosine distance -> similarity
        sims = 1.0 - neigh_dists

        # pull neighbor centered ratings for this movie
        # if neighbor hasn't rated the movie, matrix entry is 0
        neigh_centered = self.item_user_matrix[neigh_items, u_idx].toarray().ravel()

        mask = neigh_centered != 0
        if not np.any(mask):
            # no neighbor info -> item mean
            return float(self.item_means[m_idx])

        sims = sims[mask]
        neigh_centered = neigh_centered[mask]

        denom = np.sum(np.abs(sims))
        if denom < 1e-12:
            return float(self.item_means[u_idx])

        # weighted sum of centered ratings + add back item mean
        pred_centered = float(np.dot(sims, neigh_centered) / denom)
        pred = float(self.item_means[u_idx] + pred_centered)

        # optional clipping for MovieLens scale
        if pred < 0.5:
            pred = 0.5
        elif pred > 5.0:
            pred = 5.0

        return pred