import pandas as pd

# -------------------------------------------
# 1) Load both similarity matrices
# -------------------------------------------

user_based_path = "similarity_matrix.csv"                   # User-to-User similarity
item_based_path = "similarity_matrix_item_based.csv"        # Item-to-Item similarity

print("📥 Loading similarity matrices...\n")

sim_user = pd.read_csv(user_based_path, index_col=0)
sim_item = pd.read_csv(item_based_path, index_col=0)

# Fix index type (convert to int)
sim_user.index = sim_user.index.astype(int)
sim_item.index = sim_item.index.astype(int)

# Fix columns type (convert to int)
sim_user.columns = sim_user.columns.astype(int)
sim_item.columns = sim_item.columns.astype(int)

print("Available user IDs:", sim_user.index.tolist()[:20])


print("✅ User-Based Matrix Loaded:", sim_user.shape)
print("✅ Item-Based Matrix Loaded:", sim_item.shape)


# -------------------------------------------
# 2) Choose sample users & movies for analysis
# -------------------------------------------
sample_users = [1, 3, 5]          # change if needed
sample_movies = [1, 2, 3, 4, 5]   # movieId examples

print("\n🎯 Evaluating Similarities...\n")


# -------------------------------------------
# 3) Compare USER-BASED similarities
# -------------------------------------------
print("🔵 USER-BASED SIMILARITY (between users):\n")

for u1 in sample_users:
    for u2 in sample_users:
        if u1 != u2:
            similarity = sim_user.loc[u1, u2]
            print(f"👤 User {u1} ↔ User {u2} → Similarity: {similarity:.3f}")
    print("----")


# -------------------------------------------
# 4) Compare ITEM-BASED similarities
# -------------------------------------------
print("\n🟣 ITEM-BASED SIMILARITY (between movies):\n")

for m1 in sample_movies:
    for m2 in sample_movies:
        if m1 != m2:
            similarity = sim_item.loc[m1, m2]
            print(f"🎬 Movie {m1} ↔ Movie {m2} → Similarity: {similarity:.3f}")
    print("----")


# -------------------------------------------
# 5) High-level analysis (human-friendly)
# -------------------------------------------
print("\n📊 HIGH-LEVEL ANALYSIS\n")

print("""
🔵 User-based CF:
- Measures how similar two USERS are.
- Good when user ratings are dense.
- More sensitive to sparse data.
- Works better for small user sets.
- Simple and intuitive.

🟣 Item-based CF:
- Measures similarity between MOVIES.
- Much more stable (movies change slowly, users change fast).
- Industry standard (Netflix, Amazon, YouTube).
- Best for large-scale recommender systems.
- Scales better with millions of users.

""")

print("✅ Evaluation complete.")

