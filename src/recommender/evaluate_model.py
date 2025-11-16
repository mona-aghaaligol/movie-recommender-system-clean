import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ✅ اضافه کردن روت پروژه به مسیر import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT)

from src.recommender.predict_cf import predict_rating


def evaluate_cf_model(ratings_df, user_movie_matrix, similarity_df):
    """
    ارزیابی مدل CF با LOOCV روی full user-movie matrix
    """
    full_matrix = ratings_df.pivot(index="userId", columns="movieId", values="rating")

    actual_ratings = []
    predicted_ratings = []

    for user_id in full_matrix.index:
        user_ratings = full_matrix.loc[user_id].dropna()
        if len(user_ratings) < 2:
            continue

        movie_id = user_ratings.sample(1, random_state=42).index[0]
        true_rating = user_ratings[movie_id]

        if movie_id not in similarity_df.index:
            continue

        predicted = predict_rating(user_id, movie_id, user_movie_matrix, similarity_df)
        if pd.isna(predicted):
            continue

        actual_ratings.append(true_rating)
        predicted_ratings.append(predicted)

    if len(actual_ratings) == 0:
        raise ValueError("⚠️ هیچ داده کافی برای ارزیابی CF وجود ندارد.")

    mae = np.mean(np.abs(np.array(predicted_ratings) - np.array(actual_ratings)))
    rmse = np.sqrt(np.mean((np.array(predicted_ratings) - np.array(actual_ratings)) ** 2))

    return mae, rmse


if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

    ratings_df = pd.read_csv(os.path.join(base_path, "ratings.csv"))
    user_movie_matrix = pd.read_csv(os.path.join(base_path, "user_movie_matrix.csv"), index_col=0)
    similarity_df = pd.read_csv(os.path.join(base_path, "similarity_matrix.csv"), index_col=0)

    mae, rmse = evaluate_cf_model(ratings_df, user_movie_matrix, similarity_df)

    print("\n✅ Evaluation Completed")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}\n")
