"""
Movie recommender using Support Vector Machine (SVM).

Given examples of movies you like and dislike, the SVM finds a hyperplane
that separates them in the embedding space. Movies far from the hyperplane
on the "like" side are strong recommendations.
"""

import numpy as np
import pickle
from sklearn import svm
import tensorflow as tf


# Load the trained model and mappings
print("\nLoading model and data...")
model = tf.keras.models.load_model('movie_embedding_model.keras')

with open('mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

movie_to_idx = mappings['movie_to_idx']

# Extract and normalize movie embeddings
movie_weights = model.get_layer('movie_embedding').get_weights()[0]
lens = np.linalg.norm(movie_weights, axis=1)
normalized_movies = (movie_weights.T / lens).T

print(f"✓ Loaded {len(movie_to_idx)} movies")
print()

# Movies you LIKE (label = 1)
liked_movies = [
    'Star Wars: The Force Awakens',
    'The Martian (film)',
    'Tangerine (film)',
    'Straight Outta Compton (film)',
    'Fargo (film)', 
    'Brooklyn (film)',
    'Carol (film)',
    'Spotlight (film)',
    'Deadpool (film)'
]

# Movies you DISLIKE (label = 0)
disliked_movies = [
    'American Ultra',
    'The Cobbler (2014 film)',
    'Entourage (film)',
    'Fantastic Four (2015 film)',
    'Get Hard',
    'Hot Pursuit (2015 film)',
    'Mortdecai (film)',
    'Serena (2014 film)',
    'Vacation (2015 film)'
]

# Filter to movies that exist in dataset
available_liked = [m for m in liked_movies if m in movie_to_idx]
available_disliked = [m for m in disliked_movies if m in movie_to_idx]

print(f"\nLiked movies ({len(available_liked)}):")
for movie in available_liked:
    print(f"  ✓ {movie}")

print(f"\nDisliked movies ({len(available_disliked)}):")
for movie in available_disliked:
    print(f"  ✗ {movie}")

# Create training data
y = np.array([1] * len(available_liked) + [0] * len(available_disliked))
X = np.array([
    normalized_movies[movie_to_idx[movie]] 
    for movie in available_liked + available_disliked
])

print(f"\nTraining set: {len(available_liked)} liked + {len(available_disliked)} disliked = {len(y)} total")

# Train SVM
print("\n" + "="*60)
print("TRAINING SVM")
print("="*60)
print("\nFinding hyperplane that separates liked from disliked movies...")

clf = svm.SVC(kernel='linear')
clf.fit(X, y)

print("✓ SVM trained successfully")
print("\nThe SVM has learned a decision boundary in embedding space.")
print("Movies on one side are predicted as 'liked', on the other as 'disliked'.")

# Get predictions for all movies
print("\n" + "="*60)
print("GENERATING RECOMMENDATIONS")
print("="*60)
print("\nScoring all movies based on distance from hyperplane...")

# decision_function gives distance from the hyperplane
# Positive = "liked" side, Negative = "disliked" side
# Larger magnitude = more confident prediction
scores = clf.decision_function(normalized_movies)

# Sort by score
ranked_indices = np.argsort(scores)

print("✓ Ranked all movies\n")

# Show top recommendations
print("="*60)
print("TOP 10 RECOMMENDATIONS (Most Liked)")
print("="*60)
print("These movies are furthest on the 'liked' side of the hyperplane\n")

for i, idx in enumerate(reversed(ranked_indices[-10:]), 1):
    movie_title = list(movie_to_idx.keys())[idx]
    score = scores[idx]
    
    # Mark if movie was in training set
    marker = ""
    if movie_title in available_liked:
        marker = " [TRAINING: LIKED]"
    elif movie_title in available_disliked:
        marker = " [TRAINING: DISLIKED]"
    
    print(f"{i:2d}. {score:6.3f} | {movie_title}{marker}")

# Show worst predictions
print("\n" + "="*60)
print("BOTTOM 10 (Most Disliked)")
print("="*60)
print("These movies are furthest on the 'disliked' side of the hyperplane\n")

for i, idx in enumerate(ranked_indices[:10], 1):
    movie_title = list(movie_to_idx.keys())[idx]
    score = scores[idx]
    
    marker = ""
    if movie_title in available_liked:
        marker = " [TRAINING: LIKED]"
    elif movie_title in available_disliked:
        marker = " [TRAINING: DISLIKED]"
    
    print(f"{i:2d}. {score:6.3f} | {movie_title}{marker}")

