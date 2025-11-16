import numpy as np

def predict_rating(user_id, movie_id, user_movie_matrix, similarity_df):
    # 1️⃣ نمرات کاربر u
    user_ratings = user_movie_matrix.loc[user_id]
    
    # 2️⃣ فیلم‌هایی که کاربر امتیاز داده
    rated_movies = user_ratings[user_ratings.notna()].index  # S_u
    
    # اگر کاربر هیچ فیلمی ندیده باشه
    if len(rated_movies) == 0:
        return np.nan
    
    # 3️⃣ میانگین امتیازهای کاربر
    user_mean = user_ratings[rated_movies].mean()
    
    # 4️⃣ شباهت‌های فیلم i با فیلم‌های دیده‌شده
    similarities = similarity_df.loc[movie_id, rated_movies]
    
    # 5️⃣ انحراف از میانگین برای هر فیلم دیده‌شده
    rating_diffs = user_ratings[rated_movies] - user_mean
    
    # 6️⃣ محاسبه بخش وزن‌دار
    numerator = np.sum(similarities * rating_diffs)
    denominator = np.sum(np.abs(similarities))
    
    # 7️⃣ اگر مخرج صفر شد، پیش‌بینی‌ناپذیر است
    if denominator == 0:
        return user_mean  # بازگشت به میانگین کاربر
    
    # 8️⃣ پیش‌بینی نهایی
    predicted_rating = user_mean + numerator / denominator
    return predicted_rating

