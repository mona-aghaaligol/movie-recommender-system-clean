import sys
import os
import pandas as pd
import pytest

# ✅ اضافه کردن مسیر src به مسیرهای قابل دسترسی (نسبت به پوشه tests)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from recommender.domain.predict_cf import predict_rating


@pytest.fixture(scope="session")
def data():
    """Load required data once for all tests."""
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    user_movie_matrix = pd.read_csv(os.path.join(base_path, "user_movie_matrix.csv"), index_col=0)
    similarity_df = pd.read_csv(os.path.join(base_path, "similarity_matrix.csv"), index_col=0)
    users_df = pd.read_csv(os.path.join(base_path, "users.csv"), encoding="utf-8-sig")
    user_dict = dict(zip(users_df.userId, users_df.username))
    return user_movie_matrix, similarity_df, user_dict


@pytest.mark.parametrize("user_id,movie_id", [
    (1, 10),
    (1, 50),
    (1, 100),
    (3, 10),
    (3, 50),
    (5, 100),
])
def test_predict_rating_range(data, user_id, movie_id):
    """Ensure predicted ratings are valid (0 ≤ rating ≤ 5)."""
    user_movie_matrix, similarity_df, user_dict = data
    rating = predict_rating(user_id, movie_id, user_movie_matrix, similarity_df)
    assert 0 <= rating <= 5, f"❌ Invalid rating {rating:.2f} for user {user_dict[user_id]} → movie {movie_id}"
