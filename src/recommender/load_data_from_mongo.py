import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Read the Mongo connection string from .env
connection_string = os.getenv("MONGO_URI_DEV")

if not connection_string:
    raise ValueError("❌ ERROR: MONGO_URI_DEV not found in .env file!")

# Connect to MongoDB Atlas
client = MongoClient(connection_string)

# Select database and collections
db = client["movie_recommender_db"]
movies_collection = db["movies"]
ratings_collection = db["ratings"]

# Read data from MongoDB
movies_data = list(movies_collection.find({}, {"_id": 0}))
ratings_data = list(ratings_collection.find({}, {"_id": 0}))

# Convert to DataFrame
movies_df = pd.DataFrame(movies_data)
ratings_df = pd.DataFrame(ratings_data)

# Print samples
print("🎬 Movies collection sample:")
print(movies_df.head())

print("\n⭐ Ratings collection sample:")
print(ratings_df.head())

# Print shapes
print("\nMovies shape:", movies_df.shape)
print("Ratings shape:", ratings_df.shape)
