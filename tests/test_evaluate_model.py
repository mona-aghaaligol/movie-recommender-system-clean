import sys, os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from recommender.evaluate_model import evaluate_cf_model


def test_evaluate_cf_model():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

    ratings_df = pd.read_csv(os.path.join(base_path, "ratings.csv"))
    user_movie_matrix = pd.read_csv(os.path.join(base_path, "user_movie_matrix.csv"), index_col=0)
    similarity_df = pd.read_csv(os.path.join(base_path, "similarity_matrix.csv"), index_col=0)

    mae, rmse = evaluate_cf_model(ratings_df, user_movie_matrix, similarity_df)

    assert mae >= 0
    assert rmse >= 0
    assert rmse >= mae
    print(f"\n📈 MAE: {mae:.3f} | RMSE: {rmse:.3f}")

