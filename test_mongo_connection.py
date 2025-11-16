import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Read connection string
mongo_uri = os.getenv("MONGO_URI_DEV")

print("🔌 Connecting to MongoDB...")

try:
    client = MongoClient(mongo_uri)
    db_list = client.list_database_names()
    print("✅ Connected successfully!")
    print("📚 Databases:", db_list)
except Exception as e:
    print("❌ Connection failed!")
    print(e)

