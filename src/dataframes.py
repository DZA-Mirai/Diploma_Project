import pandas as pd
from sklearn.model_selection import train_test_split


def load_ratings(ratings_path: str, movies_path: str | None = None) -> pd.DataFrame:
    """
    Load MovieLens ratings (optionally merged with movies).
    """
    ratings = pd.read_csv(ratings_path)

    if movies_path is not None:
        movies = pd.read_csv(movies_path)
        ratings = ratings.merge(movies, on="movieId", how="left")

    return ratings


def split_by_user(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-user train/test split.
    """
    train_parts = []
    test_parts = []

    for _, group in df.groupby("userId"):
        train_g, test_g = train_test_split(
            group,
            test_size=test_size,
            random_state=seed,
        )
        train_parts.append(train_g)
        test_parts.append(test_g)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, test_df


def apply_warm_start_filtering(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_ratings: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only items with >= min_ratings in train and
    ensure warm-start users/items in test.
    """
    # frequent items in TRAIN
    movie_counts = train_df["movieId"].value_counts()
    valid_movies = movie_counts[movie_counts >= min_ratings].index

    train_df = train_df[train_df["movieId"].isin(valid_movies)].copy()
    test_df = test_df[test_df["movieId"].isin(valid_movies)].copy()

    # warm-start users
    valid_users = train_df["userId"].unique()
    test_df = test_df[test_df["userId"].isin(valid_users)].copy()

    # warm-start items (safety)
    train_items = train_df["movieId"].unique()
    test_df = test_df[test_df["movieId"].isin(train_items)].copy()

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_eval_subset(
    test_df: pd.DataFrame,
    n_eval: int = 200_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample validation subset from test set.
    """
    n = min(n_eval, len(test_df))
    return test_df.sample(n=n, random_state=seed).reset_index(drop=True)


def prepare_datasets(
    ratings_path: str,
    movies_path: str | None = None,
    test_size: float = 0.2,
    min_ratings: int = 5,
    eval_size: int = 200_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load → split → filter → eval subset.
    """
    df = load_ratings(ratings_path, movies_path)
    train_df, test_df = split_by_user(df, test_size, seed)
    filtered_train_df, filtered_test_df = apply_warm_start_filtering(
        train_df, test_df, min_ratings
    )
    eval_df = make_eval_subset(filtered_test_df, eval_size, seed)

    return filtered_train_df, filtered_test_df, eval_df