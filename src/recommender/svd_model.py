import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, ratings_df, epochs=10, factors=50, batch_size=256, load_from_file=False):
        super().__init__()

        self.epochs = epochs
        self.factors = factors
        self.batch_size = batch_size

        # ✅ mapping برای userId و movieId → index (برای embedding)
        self.user_to_idx = {uid: idx for idx, uid in enumerate(ratings_df["userId"].unique())}
        self.item_to_idx = {mid: idx for idx, mid in enumerate(ratings_df["movieId"].unique())}

        ratings_df["user_idx"] = ratings_df["userId"].map(self.user_to_idx)
        ratings_df["item_idx"] = ratings_df["movieId"].map(self.item_to_idx)
        self.ratings_df = ratings_df

        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)

        self.user_embedding = nn.Embedding(num_users, factors)
        self.item_embedding = nn.Embedding(num_items, factors)

        self.output_layer = nn.Linear(factors, 1)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005)

        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/svd_model.pth"))

        if load_from_file and os.path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path))
            print("✅ Pretrained SVD model loaded.")
        else:
            print("🚀 Training MF model (SVD)...")
            self.train_model()

    def train_model(self):
        users = torch.tensor(self.ratings_df["user_idx"].values, dtype=torch.long)
        items = torch.tensor(self.ratings_df["item_idx"].values, dtype=torch.long)
        ratings = torch.tensor(self.ratings_df["rating"].values, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(users, items, ratings)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.epochs + 1):
            batch_losses = []

            for user_batch, item_batch, rating_batch in loader:
                user_vec = self.user_embedding(user_batch)
                item_vec = self.item_embedding(item_batch)

                features = user_vec * item_vec
                predicted = self.output_layer(features).squeeze()

                loss = self.loss_fn(predicted, rating_batch)
                batch_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"✅ Epoch {epoch}/{self.epochs} | avg_loss={np.mean(batch_losses):.4f}")

        torch.save(self.state_dict(), self.model_path)
        print("💾 Model saved at:", self.model_path)

    def predict(self, user_id, movie_id):
        if user_id not in self.user_to_idx or movie_id not in self.item_to_idx:
            return None

        self.eval()

        user_idx = torch.tensor(self.user_to_idx[user_id])
        item_idx = torch.tensor(self.item_to_idx[movie_id])

        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)

        pred = self.output_layer(user_vec * item_vec).item()
        return max(0.5, min(5.0, pred))  # clamp score (0.5-5)


if __name__ == "__main__":
    print("📥 Loading dataset...")

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/ratings.csv"))
    ratings_df = pd.read_csv(data_path)

    print("🚀 Training SVD model...")
    model = MatrixFactorization(ratings_df, epochs=5, factors=50)

    print("✅ Training completed!")
