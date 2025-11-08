# Movie Recommendation System

A movie recommendation system built using the MovieLens dataset, which includes user ratings, tags, and movie information.

## Dataset Information

This system uses the MovieLens small dataset that contains:

- 100,836 ratings
- 3,683 tag applications
- 9,742 movies
- 610 users
- Data collected between March 29, 1996 and September 24, 2018

## Project Structure

The project contains the following files:

- `app.py` - Main application file
- `movie_recomender.py` - Core recommendation engine
- `test.py` - Test suite for the system

### Dataset Files

- `movies.csv` - Movie information including titles and genres
- `ratings.csv` - User ratings data (userId, movieId, rating, timestamp)
- `tags.csv` - User-generated movie tags
- `links.csv` - Movie identifier links
- `tmdb_5000_movies.csv` - Additional movie data from TMDB
- `tmdb_5000_credits.csv` - Movie credits data from TMDB

## Data File Structure

### Ratings (ratings.csv)

```
userId,movieId,rating,timestamp
```

- Ratings are on a 5-star scale with half-star increments (0.5 - 5.0)
- Timestamps are in UTC, seconds since January 1, 1970

### Tags (tags.csv)

```
userId,movieId,tag,timestamp
```

- Tags are user-generated metadata about movies
- Each tag is typically a single word or short phrase

## Usage License

This dataset is available for research purposes under the following conditions:

- No endorsement from the University of Minnesota or GroupLens Research Group may be stated or implied
- The dataset must be cited in any resulting publications
- The data may be redistributed under the same license conditions
- Commercial use requires permission from GroupLens Research Project

## Citation

If you use this dataset in your research, please cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

## Contact

For questions or comments, please contact: grouplens-info@umn.edu

## About GroupLens

GroupLens is a research group at the University of Minnesota's Department of Computer Science and Engineering. They specialize in:

- Recommender systems
- Online communities
- Mobile technologies
- Digital libraries
- Geographic information systems

Visit [MovieLens](http://movielens.org) to try their movie recommender system!
