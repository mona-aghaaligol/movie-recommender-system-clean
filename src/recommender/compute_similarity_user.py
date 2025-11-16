import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Read dev connection string
connection_string = os.getenv("MONGO_URI_DEV")

if not connection_string:
    raise ValueError("❌ ERROR: MONGO_URI_DEV not found in .env file!")

# Connect to MongoDB via dev user
client = MongoClient(connection_string)

# Select db and collections
db = client["movie_recommender_db"]
movies_collection = db["movies"]
ratings_collection = db["ratings"]

# Load data from MongoDB
movies_data = list(movies_collection.find({}, {"_id": 0}))
ratings_data = list(ratings_collection.find({}, {"_id": 0}))

movies_df = pd.DataFrame(movies_data)
ratings_df = pd.DataFrame(ratings_data)

# Prepare user-movie matrix
pivot_df = ratings_df.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# Compute similarity matrix using cosine similarity
similarity_matrix = cosine_similarity(pivot_df)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=pivot_df.index,
    columns=pivot_df.index
)

# Save similarity matrix
similarity_df.to_csv("similarity_matrix.csv", encoding="utf-8-sig")

print("✅ Similarity matrix created and saved as similarity_matrix.csv")
