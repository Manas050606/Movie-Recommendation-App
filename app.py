import streamlit as st
import pickle
import pandas as pd

# --- Load the Data and Model ---

# Function to load the saved files
def load_data():
    try:
        movies_df = pd.DataFrame(pickle.load(open('movies_df.pkl', 'rb')))
        cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))
        return movies_df, cosine_sim
    except FileNotFoundError:
        st.error("Error: 'movies_df.pkl' or 'cosine_sim.pkl' not found.")
        st.error("Please run the 'test2.py' script first to generate these files.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return None, None

movies_df, cosine_sim = load_data()

# We need the indices for our recommendation function
if movies_df is not None:
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

# --- Recommendation Function ---
# (This is the same function from test2.py)
def get_recommendations_v2(title, cosine_sim=cosine_sim, num_recommendations=10):
    try:
        idx = indices[title]
    except KeyError:
        return "Movie not found"
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

# --- Streamlit App Layout ---

st.title('🎬 Movie Recommender System')
st.markdown('Find movies similar to your favorites!')

if movies_df is not None:
    # Create a dropdown select box
    selected_movie = st.selectbox(
        'Type or select a movie you like:',
        movies_df['title'].values
    )
    lang_filter = st.checkbox('Show movies in the same language only', value=True)

    # Add a "Recommend" button
    # Add a "Recommend" button
if st.button('Get Recommendations'):
    if selected_movie:
        with st.spinner('Finding similar movies...'):
            # Get the list of recommended movie TITLES
            recommendations_titles = get_recommendations_v2(selected_movie)

            # Get the full DataFrame for these recommended movies
            rec_df = movies_df[movies_df['title'].isin(recommendations_titles)]

            final_recommendations = []

            # Check if the filter is on
            if lang_filter:
                # 1. Get the language of the movie we selected
                source_lang = movies_df[movies_df['title'] == selected_movie].original_language.iloc[0]

                # 2. Filter our recommendations to only include movies with that same language
                final_df = rec_df[rec_df['original_language'] == source_lang]

                # 3. Get the titles from this filtered list
                final_recommendations = final_df['title']
            else:
                # If filter is off, just use all the original recommendations
                final_recommendations = recommendations_titles


            st.subheader(f'Top movies similar to "{selected_movie}":')

            # Display the list of movies
            if not final_recommendations.empty:
                for i, movie in enumerate(final_recommendations):
                    st.write(f"{i+1}. {movie}")
            else:
                st.write("No recommendations found with the selected filter.")
    else:
        st.warning("Please select a movie.")