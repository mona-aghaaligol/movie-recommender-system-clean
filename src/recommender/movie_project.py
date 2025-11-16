import os
from pymongo import MongoClient
from dotenv import load_dotenv

# -----------------------------------
# 1) Load environment variables
# -----------------------------------
load_dotenv()

connection_string = os.getenv("MONGO_URI_DEV")

if not connection_string:
    raise ValueError("❌ ERROR: MONGO_URI_DEV not found in .env file!")

# -----------------------------------
# 2) Connect to MongoDB
# -----------------------------------
client = MongoClient(connection_string)
db = client["movie_recommender_db"]
movies_collection = db["movies"]

# -----------------------------------
# 3) Sample insert (for testing only)
# -----------------------------------
sample_movie = {
    "movieId": 999999,
    "title": "Sample Movie For Test",
    "genres": ["Test"]
}

result = movies_collection.insert_one(sample_movie)
print("✅ Inserted document ID:", result.inserted_id)
