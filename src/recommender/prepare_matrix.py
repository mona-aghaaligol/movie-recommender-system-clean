import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# -------------------------------------
# 1) Load .env file
# -------------------------------------
load_dotenv()

connection_string = os.getenv("MONGO_URI_DEV")
if not connection_string:
    raise ValueError("❌ ERROR: MONGO_URI_DEV not found in .env")

# -------------------------------------
# 2) Connect to MongoDB
# -------------------------------------
client = MongoClient(connection_string)
db = client["movie_recommender_db"]

# Load movies and ratings
movies = pd.DataFrame(list(db.movies.find({}, {"_id": 0})))
ratings = pd.DataFrame(list(db.ratings.find({}, {"_id": 0})))

# -------------------------------------
# 3) Build the user–movie matrix
# -------------------------------------
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# -------------------------------------
# 4) Save the matrix locally
# -------------------------------------
user_movie_matrix.to_csv("user_movie_matrix.csv", encoding="utf-8-sig")

print("✅ User-Movie Matrix ساخته شد و روی دیسک ذخیره شد!")
print(user_movie_matrix.head())
print(f"\nShape: {user_movie_matrix.shape}")
