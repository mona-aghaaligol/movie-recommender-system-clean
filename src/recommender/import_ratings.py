import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Read dev connection string
connection_string = os.getenv("MONGO_URI_DEV")

if not connection_string:
    raise ValueError("❌ ERROR: MONGO_URI_DEV not found in .env file!")

# Connect to MongoDB via dev user
client = MongoClient(connection_string)

# Select db and collection
db = client["movie_recommender_db"]
ratings_collection = db["ratings"]

# Load data from JSON file
with open("ratings.json", "r", encoding="utf-8") as f:
    ratings_data = json.load(f)

# Insert into MongoDB
result = ratings_collection.insert_many(ratings_data)

print(f"✅ تعداد {len(result.inserted_ids)} داده به کالکشن ratings اضافه شد.")
