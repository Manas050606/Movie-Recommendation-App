import pandas as pd

# Make sure to update this path to where your file is located
# e.g., 'ml-latest-small/movies.csv'
try:
    movies_df = pd.read_csv('movies.csv')
    print("File loaded successfully!")
    print(movies_df.head())
except FileNotFoundError:
    print("Error: 'movies.csv' not found.")
    print("Please download the 'ml-latest-small' dataset and place it in the correct folder.")
    # --- Step 2: Prepare Genres ---

# Replace the pipe '|' with a space ' '
movies_df['genres'] = movies_df['genres'].str.replace('|', ' ', regex=False)

# Handle movies with no genres listed
movies_df['genres'] = movies_df['genres'].replace('(no genres listed)', '')

print("\nGenres cleaned:")
print(movies_df.head())

from sklearn.feature_extraction.text import TfidfVectorizer

# --- Step 3: Vectorize Genres ---

# Create a TF-IDF Vectorizer
# stop_words='english' will remove common words, which isn't
# strictly necessary for genres but is good practice.
tfidf = TfidfVectorizer(stop_words='english')

# Create the TF-IDF matrix by fitting and transforming the genres data
# This matrix will show the "score" of each genre for each movie
genre_matrix = tfidf.fit_transform(movies_df['genres'])

# Print the shape of the matrix
# It should be (number_of_movies, number_of_unique_genres)
print(f"\nShape of TF-IDF matrix: {genre_matrix.shape}")

from sklearn.metrics.pairwise import cosine_similarity

# --- Step 4: Calculate Similarity ---

# Compute the cosine similarity matrix
# This compares every movie with every other movie
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

print(f"\nShape of Cosine Similarity matrix: {cosine_sim.shape}")
print("\nExample similarity scores (first 5 movies):")
print(cosine_sim[0:5, 0:5])


# --- Step 5: Build the Recommender Function ---

# Create a 'Series' that has movie titles as the index 
# and the dataframe index as the value. This allows us
# to easily look up a movie's index by its title.
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, num_recommendations=10):
    """
    Finds the top N most similar movies based on genre.
    """
    # 1. Get the index of the movie that matches the title
    try:
        idx = indices[title]
    except KeyError:
        return f"Error: Movie '{title}' not found in the dataset."

    # 2. Get the pairwise similarity scores of all movies with that movie
    # This is just a single row from our cosine_sim matrix
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 3. Sort the movies based on the similarity scores (in descending order)
    # We sort by the second element (x[1]) of each tuple (index, score)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 4. Get the scores of the top N most similar movies
    # We skip the first one (index 1) because it's the movie itself
    sim_scores = sim_scores[1 : num_recommendations + 1]

    # 5. Get the movie indices from the sorted list
    movie_indices = [i[0] for i in sim_scores]

    # 6. Return the titles of the top N movies
    return movies_df['title'].iloc[movie_indices]

# --- Let's test it! ---
movie_title = "Toy Story (1995)"
print(f"\n--- Recommendations for '{movie_title}' ---")
print(get_recommendations(movie_title))

print("\n--- Recommendations for 'Jumanji (1995)' ---")
print(get_recommendations("Jumanji (1995)"))

print("\n--- Recommendations for 'Pulp Fiction (1994)' ---")
print(get_recommendations("Pulp Fiction (1994)"))