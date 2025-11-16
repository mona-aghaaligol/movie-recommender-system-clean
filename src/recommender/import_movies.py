import os
import csv
from pymongo import MongoClient
from dotenv import load_dotenv

# ----------------------------
# 0️⃣ Load environment variables
# ----------------------------
load_dotenv()

# Read dev connection string
connection_string = os.getenv("MONGO_URI_DEV")

if not connection_string:
    raise ValueError("❌ ERROR: MONGO_URI_DEV not found in .env file!")

# ----------------------------
# 1️⃣ Connect to MongoDB Atlas with dev user
# ----------------------------
client = MongoClient(connection_string)
db = client["movie_recommender_db"]
movies_collection = db["movies"]

# ----------------------------
# 2️⃣ مسیر فایل CSV
# ----------------------------
csv_file_path = "movies.csv"

# ----------------------------
# 3️⃣ خواندن CSV و اضافه کردن فیلم‌ها
# ----------------------------
with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for movie in reader:
        if movies_collection.find_one({"movieId": int(movie["movieId"])}):
            print(f"⚠️ Movie with ID {movie['movieId']} already exists. Skipping.")
        else:
            new_movie = {
                "movieId": int(movie["movieId"]),
                "title": movie["title"],
                "genres": movie["genres"].split("|")
            }
            result = movies_collection.insert_one(new_movie)
            print(f"✅ Added movie: {new_movie['title']} (ObjectID: {result.inserted_id})")

print("✅ Import completed.")
