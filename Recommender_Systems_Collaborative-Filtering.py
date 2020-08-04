# %%
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# %% Load Data
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
# Dropping the genres column
movies_df = movies_df.drop('genres', 1)

print(movies_df.head())

# ratings dataframe
print(ratings_df.head())
# Every row in the ratings dataframe has a user id associated with at least one movie, a rating and
# a timestamp showing when they reviewed it
# drop timestamp column as it is not needed
ratings_df = ratings_df.drop('timestamp', 1)
print(ratings_df.head())

# %% Collaborative Filtering/User-User Filtering
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
inputMovies = inputMovies.drop('year', 1)

print(inputMovies)

# Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
print(userSubset.head())

# group rows by user ID
userSubsetGroup = userSubset.groupby(['userId'])
print(userSubsetGroup.get_group(1130))

# Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)
print(userSubsetGroup[:3])

# %% Similarity of users to input user using Pearson Correlation Coefficient
# select a subset of users to iterate through
userSubsetGroup = userSubsetGroup[0:100]

# Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

# For every user group in our subset
for name, group in userSubsetGroup:
    # sort the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    # Get the N for the formula
    nRatings = len(group)
    # Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    # store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    # put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    # Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
    Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
    Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(tempGroupList) / float(
        nRatings)

    # If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
    else:
        pearsonCorrelationDict[name] = 0

print(pearsonCorrelationDict.items())

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
print(pearsonDF.head())

# top x similar users to input user
topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[:50]
print(topUsers.head())

# %% Recommend to User
"""
Rating of selected users to all movies:
Take the weighted average of the ratings of the movies using the Pearson Correlation as the weight, but how?
Get the movies watched by the users in our pearsonDF from the ratings dataframe and then store their correlation 
in a new column called _similarityIndex". 
This is achieved below by merging of these two tables.
"""
topUsersRating = topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
print(topUsersRating.head())

# Multiply the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * topUsersRating['rating']
print(topUsersRating.head())

# Apply sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
print(tempTopUsersRating.head())

# Create an empty dataframe
recommendation_df = pd.DataFrame()
# take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating'] / \
                                                             tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
print(recommendation_df.head())

recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
print(recommendation_df.head(10))
# %%
movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
