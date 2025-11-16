import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.recommender.svd_model import MatrixFactorization

if __name__ == "__main__":
    print("📥 Loading dataset...")

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/ratings.csv"))
    ratings_df = pd.read_csv(data_path)

    print("🔁 Loading trained SVD model...")
    model = MatrixFactorization(ratings_df, epochs=0, load_from_file=True)

    print("🔍 Evaluating model...")

    test_df = ratings_df.sample(5000, random_state=42)

    actual = []
    predicted = []

    for _, row in test_df.iterrows():
        pred = model.predict(row["userId"], row["movieId"])
        if pred is None:
            continue

        actual.append(row["rating"])
        predicted.append(pred)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print("\n✅ SVD Evaluation Completed")
    print("MAE: ", round(mae, 4))
    print("RMSE:", round(rmse, 4))
