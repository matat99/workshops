"""
Find similar movies using learned embeddings.

Movies with similar Wikipedia links have similar embeddings,
so we can find neighbors by computing distances in embedding space.
"""

import numpy as np
import pickle
import tensorflow as tf


# Load the trained model and mappings
print("\nLoading model and data...")
model = tf.keras.models.load_model('movie_embedding_model.keras')

with open('mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

movie_to_idx = mappings['movie_to_idx']
all_movies = mappings['all_movies']

print(f"✓ Loaded model with {len(movie_to_idx)} movies")

# Extract and normalize movie embeddings
print("Extracting embeddings...")
movie_weights = model.get_layer('movie_embedding').get_weights()[0]
lens = np.linalg.norm(movie_weights, axis=1)
normalized = (movie_weights.T / lens).T

print(f"✓ Each movie represented as {movie_weights.shape[1]}-dimensional vector")
print()

def find_similar_movies(movie_title, top_n=10):
    """
    Find movies most similar to the given movie.
    
    Args:
        movie_title: Name of the movie to find neighbors for
        top_n: Number of similar movies to return
    
    Returns:
        List of (movie_name, similarity_score) tuples
    """
    if movie_title not in movie_to_idx:
        print(f"❌ '{movie_title}' not found in dataset")
        return []
    
    # Get the movie's embedding
    movie_idx = movie_to_idx[movie_title]
    movie_vector = normalized[movie_idx]
    
    # Compute cosine similarity with all movies
    similarities = np.dot(normalized, movie_vector)
    
    # Get top N most similar (including the movie itself)
    most_similar_indices = np.argsort(similarities)[-(top_n + 1):]
    
    results = []
    for idx in reversed(most_similar_indices):
        similar_movie_title = all_movies[idx][0]
        similarity_score = similarities[idx]
        results.append((similar_movie_title, similarity_score))
    
    return results

test_movies = [
    'Rogue One',
    'Fargo',
    'Deadpool (film)'
]

for movie in test_movies:
    print(f"\nMovies similar to '{movie}':")
    print("-" * 60)
    
    similar = find_similar_movies(movie, top_n=10)
    
    if similar:
        for title, score in similar:
            # Mark the query movie with an arrow
            marker = "→ " if title == movie else "  "
            print(f"{marker}{score:.3f} | {title}")
