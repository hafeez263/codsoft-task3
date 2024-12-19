import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
data = {
    'User': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'],
    'Inception': [5, 4, 0, 5, 4],
    'Titanic': [4, 0, 5, 4, 4],
    'Avatar': [5, 0, 4, 4, 0],
    'Avengers': [0, 5, 4, 0, 5],
    'Joker': [0, 5, 4, 0, 4]
}
df = pd.DataFrame(data)
df.set_index('User', inplace=True)

df.fillna(0, inplace=True)


similarity_matrix = cosine_similarity(df)
similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

print("User Similarity Matrix:")
print(similarity_df)

def recommend_movies(user, df, similarity_df):
    user_ratings = df.loc[user]
    similar_users = similarity_df[user].sort_values(ascending=False)
    recommended_movies = {}

    for other_user, similarity_score in similar_users.items():
        if other_user == user:
            continue
        for movie, rating in df.loc[other_user].items():
            if user_ratings[movie] == 0 and rating > 0:
                if movie not in recommended_movies:
                    recommended_movies[movie] = similarity_score * rating
                else:
                    recommended_movies[movie] += similarity_score * rating
    
   
    recommended_movies = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, score in recommended_movies]

print("\nRecommended Movies for Alice:")
print(recommend_movies('Alice', df, similarity_df))
