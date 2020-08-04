# %%
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# %% Loading
# Dataset Link
# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip
movies_df = pd.read_csv('ml-latest/movies.csv')
ratings_df = pd.read_csv('ml-latest/ratings.csv')

print(movies_df.head())
print(ratings_df.head())

# %% Preprocessing
# remove the year from the title column by using pandas' replace function and store in a new year column

# Using regular expressions to find a year stored between parentheses
# We specify the parentheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
# Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
# Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
# Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

print(movies_df.head())

# split the values in the Genres column into a list of Genres to simplify future use
# Every genre is separated by a | so we simply have to call the split function on |
movies_df['genres'] = movies_df.genres.str.split('|')
print(movies_df.head())

# One Hot Encoding technique to convert the list of genres to a vector
# Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies_df.copy()
# For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in moviesWithGenres_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
# Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
print(moviesWithGenres_df.head())

# ratings dataframe
print(ratings_df.head())
# Every row in the ratings dataframe has a user id associated with at least one movie, a rating and
# a timestamp showing when they reviewed it
# drop timestamp column as it is not needed
ratings_df = ratings_df.drop('timestamp', 1)
print(ratings_df.head())

# %% Content-Based Recommendation System
# create an input user to recommend movies to
userInput = [
    {'title': 'The Breakfast Club', 'rating': 5},
    {'title': 'Toy Story', 'rating': 3.5},
    {'title': 'Jumanji', 'rating': 2},
    {'title': 'Pulp Fiction', 'rating': 5},
    {'title': 'Akira', 'rating': 4.5}
]
inputMovies = pd.DataFrame(userInput)
print(inputMovies)

# add movieid to input user
# filter out the rows that contain the input movie's title and then merge this subset with the input dataframe
# Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
# Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
# Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

print(inputMovies)

# get the subset of movies that the input has watched from the Dataframe containing genres defined with binary value
# Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
print(userMovies)

# Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
# Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(userGenreTable)

# using the input's reviews and multiplying them into the input's genre table and
# then summing up the resulting table by column
print(inputMovies['rating'])

# Dot product to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
# The user profile
print(userProfile)

# %% recommend movies to user
# Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
# drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(genreTable.head())
print(genreTable.shape)

# Multiply the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable * userProfile).sum(axis=1)) / (userProfile.sum())
print(recommendationTable_df.head())
# Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
# Just a peek at the values
print(recommendationTable_df.head())

# %%The final recommendation table
movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]

