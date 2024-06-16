import pandas as pd

########################
# User Based Recommender
########################

# Reading and inspecting datasets.
main_movie_df = pd.read_csv('Datasets/movie.csv')
main_rating_df = pd.read_csv('Datasets/rating.csv')
df_m = main_movie_df.copy()
df_r = main_rating_df.copy()

df_r.head()
df_m.head()

# Merging datasets.
df_r = df_r.merge(df_m, how='left', on='movieId')

# Selecting and removing movies that have less than 1000 votes.
movies_to_remove = df_r.groupby('movieId').agg({'rating': 'count'}).reset_index()
movies_to_remove = movies_to_remove[movies_to_remove['rating'] < 1000]
df_r = df_r[~df_r['movieId'].isin(movies_to_remove['movieId'])]

# Creating pivot table to see every user' ratings for every movie.
df_r.pivot_table(index=['userId'], columns=['title'], values='rating')

# Function for easy pivot table creation.
def create_user_rating(type='title', rating=1000):
    df_m = pd.read_csv('Datasets/movie.csv')
    df_r = pd.read_csv('Datasets/rating.csv')
    df_r = df_r.merge(df_m, how='left', on='movieId')
    movies_to_remove = df_r.groupby('movieId').agg({'rating': 'count'}).reset_index()
    movies_to_remove = movies_to_remove[movies_to_remove['rating'] < rating]
    df_r = df_r[~df_r['movieId'].isin(movies_to_remove['movieId'])]
    return df_r.pivot_table(index=['userId'], columns=[type], values='rating')

user_rating_table = create_user_rating()

# Random user for user-based analysis.
random_user = int(pd.Series(user_rating_table.index).sample(1).values)

# Random user' dataframe.
random_user_df = user_rating_table[user_rating_table.index == random_user]

# Selecting movies that random user watched.
movies_watch = random_user_df.columns[random_user_df.notna().any()].tolist()

# Creating new dataframe with only watched movies.
movies_watch_df = user_rating_table.loc[:, movies_watch]

# Finding out how many user watched at least 1 in listed movies.
user_movie_count = movies_watch_df.T.notnull().sum().reset_index().rename(columns={0: 'count'})

# Selecting users that watched movies at least %60 from list
user_same_movies = movies_watch_df[movies_watch_df.index.isin(user_movie_count[user_movie_count['count'] > len(movies_watch)*60/100]['userId'])]

# Copy final dataframe as backup.
final_df = user_same_movies.copy()

# Correlation of every user.
corr_df = final_df.T.corr().unstack().sort_values(ascending=False).drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=['corr'])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df.reset_index(inplace=True)

# Selecting users that has higher than %65 correlation rate with our random user.
top_users = corr_df[(corr_df['user_id_1'] == random_user) & (corr_df['corr'] >= 0.65)][['user_id_2', 'corr']].reset_index(drop=True)
top_users.columns = ['userId', 'corr']

# Reading ratings dataframe and merging it.
main_rating_df = pd.read_csv('Datasets/rating.csv')
df_r = main_rating_df.copy()
top_users_df = top_users.merge(df_r, how='left', on='userId')
del main_rating_df, df_r

# Creating weighted score.
top_users_df['weighted_rating'] = top_users_df['corr'] * top_users_df['rating']

# Creating final recommendation dataframe.
recommendation_df = top_users_df.groupby('movieId').agg({'rating': 'mean'})

# Selecting ratings that higher than 3.5 / 5 and sorting them.
recommendation_df = recommendation_df[recommendation_df['rating'] >= 3.5].sort_values('rating', ascending=False)

# Reading movies dataframe and merging it.
main_movie_df = pd.read_csv('Datasets/movie.csv')
df_m = main_movie_df.copy()
recommendation_df = recommendation_df.merge(df_m, how='left', on='movieId')
del main_movie_df, df_m

# Recommendation test.
movies_to_recommend = recommendation_df.iloc[:5]['title'].tolist()


############
# Item Based
############

# Reading and inspecting dataframes.
main_movie_df = pd.read_csv('Datasets/movie.csv')
main_rating_df = pd.read_csv('Datasets/rating.csv')
df_m = main_movie_df.copy()
df_r = main_rating_df.copy()

df_m.head()
df_r.head()
df_r.info()

# Converting 'timestamp' column' type to datetime.
df_r['timestamp'] = pd.to_datetime(df_r['timestamp'])

# Selecting random user and finding his/her best and last rating.
random_user = 1
latest = df_r[(df_r['userId'] == random_user) & (df_r['rating'] == 5)]['timestamp'].max() # YA İKİ FİLME AYNI ANDA NASIL OY VEREBİLİYORSUN
film_id = df_r[(df_r['userId'] == random_user) & (df_r['rating'] == 5) & (df_r['timestamp'] == latest)]['movieId']
film_id = user_rating_table[film_id]

# Creating movies pivot table with movie IDs.
user_rating_table = create_user_rating(type='movieId', rating=1000)

# Movie recommendation with user' correlation with every user.
recommended_movies = user_rating_table.corrwith(film_id).sort_values(ascending=False).head(10)

# Recommendation
recommended_movies.iloc[1:6]
