import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------
# 1) Load environment variables (.env)
# -------------------------------------
load_dotenv()

connection_string = os.getenv("MONGO_URI_DEV")

if not connection_string:
    raise ValueError("❌ ERROR: MONGO_URI_DEV not found in .env file!")

# -------------------------------------
# 2) Connect to MongoDB (dev user)
# -------------------------------------
client = MongoClient(connection_string)
db = client["movie_recommender_db"]

# Collections
movies_collection = db["movies"]
ratings_collection = db["ratings"]

# -------------------------------------
# 3) Load data
# -------------------------------------
ratings_data = list(ratings_collection.find({}, {"_id": 0}))
ratings_df = pd.DataFrame(ratings_data)

print("📥 Data loaded from MongoDB.")
print(ratings_df.head())

# -------------------------------------
# 4) Create user-movie matrix
# -------------------------------------
pivot_df = ratings_df.pivot(index="userId", columns="movieId", values="rating")

# Fill NaN with 0 (standard for CF baseline models)
pivot_filled = pivot_df.fillna(0)

# -------------------------------------
# 5) ITEM-BASED CF → similarity between movies (columns)
# -------------------------------------
similarity_matrix = cosine_similarity(pivot_filled.T)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=pivot_filled.columns,     # movie IDs
    columns=pivot_filled.columns    # movie IDs
)

# -------------------------------------
# 6) Save output
# -------------------------------------
output_path = "similarity_matrix_item_based.csv"
similarity_df.to_csv(output_path, encoding="utf-8-sig")

print("🎬 Item-based similarity matrix created!")
print("📁 Saved to:", output_path)
print("🔍 Preview:")
print(similarity_df.iloc[:5, :5])
