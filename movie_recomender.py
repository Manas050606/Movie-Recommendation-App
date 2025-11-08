import pandas as pd
import numpy as np
import pickle

# --- Step 1: Load and Merge Data ---

try:
    movies_df = pd.read_csv('tmdb_5000_movies.csv')
    credits_df = pd.read_csv('tmdb_5000_credits.csv')
except FileNotFoundError:
    print("Error: 'tmdb_5000_movies.csv' or 'tmdb_5000_credits.csv' not found.")
    print("Please download the 'TMDB 5000 Movie Dataset' from Kaggle.")
    exit()


# The 'credits' file has a 'movie_id' column,
# but the 'movies' file has an 'id' column. Let's rename it to match.
movies_df.rename(columns={'id': 'movie_id'}, inplace=True)

# Merge the two dataframes on the 'movie_id' column
df = credits_df.merge(movies_df, on='movie_id')

# Let's see what we have
print("Merged data shape:", df.shape)

# We don't need all these columns. Let's select just the ones we want.
features = ['movie_id', 'title_y', 'overview', 'genres', 'keywords', 'cast', 'crew', 'original_language']
df_filtered = df[features]

# Let's rename 'title_y' back to 'title' for simplicity
df_filtered = df_filtered.rename(columns={'title_y': 'title'})

print("\nSuccessfully selected and renamed columns:")
print(df_filtered.head())

import ast # To safely parse the string-formatted lists

# --- Step 2: Clean and Process the Features ---

# Let's use the df_filtered from now on
df = df_filtered

# Handle missing data in 'overview'
df['overview'] = df['overview'].fillna('')

# This function will convert the string-list into a real list
# and extract the 'name' from each dictionary
def extract_names(obj):
    """
    Parses a string-list of dictionaries and returns a list of 'name' values.
    Example: '[{"id": 28, "name": "Action"}]' -> ['Action']
    """
    try:
        # Safely evaluate the string as a Python list
        obj_list = ast.literal_eval(obj)
        names = [i['name'] for i in obj_list]
        return names
    except (ValueError, SyntaxError):
        # In case of bad data (e.g., an empty or malformed string)
        return []

# This function is similar, but only takes the first 3 names (main cast)
def extract_top_3_cast(obj):
    try:
        obj_list = ast.literal_eval(obj)
        names = [i['name'] for i in obj_list]
        return names[:3] # Get only the top 3
    except (ValueError, SyntaxError):
        return []

# This function searches the 'crew' list to find the Director
def extract_director(obj):
    try:
        obj_list = ast.literal_eval(obj)
        for i in obj_list:
            if i['job'] == 'Director':
                return [i['name']] # Return as a list for consistency
        return [] # If no director is found
    except (ValueError, SyntaxError):
        return []

# --- Apply the functions to our columns ---

# This might show a SettingWithCopyWarning, which is okay for this project.
pd.options.mode.chained_assignment = None # Suppress the warning

print("\nProcessing 'genres', 'keywords', 'cast', and 'crew'...")

df['genres'] = df['genres'].apply(extract_names)
df['keywords'] = df['keywords'].apply(extract_names)
df['cast'] = df['cast'].apply(extract_top_3_cast)
df['crew'] = df['crew'].apply(extract_director)

print("Processing complete!")
print("\nData after processing:")
print(df.head())
# --- Step 3: Create the 'tags' Column ---

# Function to remove spaces from names
def remove_spaces(words_list):
    return [i.replace(" ", "") for i in words_list]

print("\nRemoving spaces from names...")
df['genres'] = df['genres'].apply(remove_spaces)
df['keywords'] = df['keywords'].apply(remove_spaces)
df['cast'] = df['cast'].apply(remove_spaces)
df['crew'] = df['crew'].apply(remove_spaces)

# 'overview' is a string, not a list. We need to split it.
df['overview'] = df['overview'].apply(lambda x: x.split())

print("Combining all features into a 'tags' column...")
# Combine all the feature lists into one single list
df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']

# We no longer need the original columns, just 'movie_id', 'title', and 'tags'
final_df = df[['movie_id', 'title', 'tags', 'original_language']]

# The 'tags' column is currently a list. Let's join it into a single string.
final_df['tags'] = final_df['tags'].apply(lambda x: " ".join(x))

# Convert to lowercase, which is a best practice for vectorizers
final_df['tags'] = final_df['tags'].str.lower()

print("Final DataFrame created!")
print(final_df.head())

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Step 4: Vectorize the 'tags' ---

# Initialize the CountVectorizer
# max_features=5000 means we'll take the 5000 most frequent words/tags
# stop_words='english' removes 'and', 'the', 'of', etc.
cv = CountVectorizer(max_features=5000, stop_words='english')

print("\nVectorizing tags...")
# fit_transform converts the text into a numeric matrix
# We use .toarray() to make it a full (dense) matrix
vectors = cv.fit_transform(final_df['tags']).toarray()

print(f"Shape of vectors: {vectors.shape}")

# --- Step 5: Calculate Similarity ---

print("Calculating cosine similarity...")
cosine_sim = cosine_similarity(vectors)

print(f"Shape of similarity matrix: {cosine_sim.shape}")

# --- Step 6: Build the Recommender Function ---

# We still need a way to look up a movie's index from its title
indices = pd.Series(final_df.index, index=final_df['title']).drop_duplicates()

def get_recommendations_v2(title, cosine_sim=cosine_sim, num_recommendations=10):
    """
    Finds the top N most similar movies based on 'tags'.
    """
    # 1. Get the index of the movie that matches the title
    try:
        idx = indices[title]
    except KeyError:
        return f"Error: Movie '{title}' not found in the dataset."

    # 2. Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 3. Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 4. Get the scores of the top N most similar movies
    sim_scores = sim_scores[1 : num_recommendations + 1]

    # 5. Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # 6. Return the titles
    return final_df['title'].iloc[movie_indices]

# --- Let's test our new system! ---

# Try the same movie as last time and see the difference
movie_title = "The Dark Knight Rises"
print(f"\n--- Recommendations for '{movie_title}' ---")
print(get_recommendations_v2(movie_title))

print("\n--- Recommendations for 'Avatar' ---")
print(get_recommendations_v2("Avatar"))

print("\n--- Recommendations for 'Spectre' ---")
print(get_recommendations_v2("Spectre"))

# --- Step 7: Save Model and Data for the App ---

print("\nSaving data for Streamlit app...")

# Save the final dataframe
pickle.dump(final_df, open('movies_df.pkl', 'wb'))

# Save the similarity matrix
pickle.dump(cosine_sim, open('cosine_sim.pkl', 'wb'))

print("Data saved! You can now run the Streamlit app.")