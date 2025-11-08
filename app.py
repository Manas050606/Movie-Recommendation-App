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

    # Add a "Recommend" button
    if st.button('Get Recommendations'):
        if selected_movie:
            with st.spinner('Finding similar movies...'):
                recommendations = get_recommendations_v2(selected_movie)
                st.subheader(f'Top 10 movies similar to "{selected_movie}":')
                
                # Display the list of movies
                for i, movie in enumerate(recommendations):
                    st.write(f"{i+1}. {movie}")
        else:
            st.warning("Please select a movie.")
else:
    st.info("Waiting for data files to be loaded...")