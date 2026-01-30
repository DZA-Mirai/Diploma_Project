import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse_mae(y_true, y_pred):
    """
    Compute RMSE and MAE.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def evaluate_df(model, df):
    """
    Evaluate a model on a DataFrame with columns:
    userId, movieId, rating
    """
    y_true = df["rating"].to_numpy()
    y_pred = np.array([
        model.predict(u, i)
        for u, i in zip(df["userId"], df["movieId"])
    ])

    return rmse_mae(y_true, y_pred)