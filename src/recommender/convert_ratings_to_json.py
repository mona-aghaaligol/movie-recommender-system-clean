import csv
import json

# مسیر فایل ورودی (CSV)
csv_file = "ratings.csv"

# مسیر فایل خروجی (JSON)
json_file = "ratings.json"

# لیستی برای ذخیره داده‌ها
data = []

# خواندن فایل CSV و تبدیل به دیکشنری
with open(csv_file, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # تبدیل مقادیر عددی به نوع مناسب
        row["userId"] = int(row["userId"])
        row["movieId"] = int(row["movieId"])
        row["rating"] = float(row["rating"])
        row["timestamp"] = int(row["timestamp"])
        data.append(row)

# نوشتن داده‌ها در فایل JSON
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("✅ فایل ratings.json با موفقیت ساخته شد!")

