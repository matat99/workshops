"""
Train movie embeddings using Wikipedia link co-occurrence.

The model learns to predict whether a movie links to a Wikipedia page.
As a byproduct, movies with similar links end up with similar embeddings.
"""

import json
from collections import Counter
import tensorflow as tf
import numpy as np
import random
import pickle

print("="*60)
print("TRAINING MOVIE EMBEDDINGS")
print("="*60)

# Load and process data
print("\n1. Loading movie data...")
all_movies = []
link_counts = Counter()

with open('wp_movies_10k.ndjson', 'r') as f:
    for i, line in enumerate(f):
        movie = json.loads(line)
        all_movies.append(movie)
        link_counts.update(movie[2])  # movie[2] contains Wikipedia links
        
        if (i + 1) % 2000 == 0:
            print(f"   Processed {i + 1} movies...")

print(f"✓ Loaded {len(all_movies)} movies")
print(f"✓ Found {len(link_counts)} unique links")

# Filter to frequently occurring links
print("\n2. Creating mappings...")
top_links = [link for link, count in link_counts.items() if count >= 3]
link_to_idx = {link: idx for idx, link in enumerate(top_links)}
movie_to_idx = {movie[0]: idx for idx, movie in enumerate(all_movies)}

print(f"✓ Kept {len(top_links)} frequent links (appear in 3+ movies)")
print(f"✓ Mapped {len(movie_to_idx)} movies to indices")

# Create training pairs (movie, link) where movie actually links to that page
print("\n3. Creating training pairs...")
pairs = []
for movie in all_movies:
    if movie[0] in movie_to_idx:
        movie_idx = movie_to_idx[movie[0]]
        for link in movie[2]:
            if link in link_to_idx:
                pairs.append((link_to_idx[link], movie_idx))

pairs_set = set(pairs)
print(f"✓ Created {len(pairs):,} positive training pairs")

# Build the model
print("\n4. Building neural network...")

def create_embedding_model(embedding_size=30):
    """
    Creates a model that learns embeddings by predicting link co-occurrence.
    
    Architecture:
    - Two embedding layers (one for movies, one for links)
    - Dot product to measure similarity
    - Output: probability of co-occurrence
    """
    link_input = tf.keras.layers.Input(name='link', shape=(1,))
    movie_input = tf.keras.layers.Input(name='movie', shape=(1,))
    
    link_embedding = tf.keras.layers.Embedding(
        name='link_embedding',
        input_dim=len(top_links),
        output_dim=embedding_size
    )(link_input)
    
    movie_embedding = tf.keras.layers.Embedding(
        name='movie_embedding',
        input_dim=len(movie_to_idx),
        output_dim=embedding_size
    )(movie_input)
    
    # Dot product measures similarity
    dot = tf.keras.layers.Dot(name='dot_product', normalize=True, axes=2)(
        [link_embedding, movie_embedding]
    )
    output = tf.keras.layers.Reshape((1,))(dot)
    
    model = tf.keras.models.Model(inputs=[link_input, movie_input], outputs=[output])
    model.compile(optimizer='nadam', loss='mse')
    
    return model

model = create_embedding_model()
print("✓ Model created")
model.summary()

# Training data generator
def generate_batches(pairs, positive_samples=50, negative_ratio=5):
    """
    Generates training batches with positive and negative examples.
    
    Positive: Real movie-link pairs (label = 1)
    Negative: Random pairs that don't exist (label = -1)
    """
    batch_size = positive_samples * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    while True:
        # Add positive samples
        for idx, (link_id, movie_id) in enumerate(random.sample(pairs, positive_samples)):
            batch[idx, :] = (link_id, movie_id, 1)
        
        # Add negative samples
        idx = positive_samples
        while idx < batch_size:
            movie_id = random.randrange(len(movie_to_idx))
            link_id = random.randrange(len(top_links))
            if (link_id, movie_id) not in pairs_set:
                batch[idx, :] = (link_id, movie_id, -1)
                idx += 1
        
        np.random.shuffle(batch)
        yield {'link': batch[:, 0], 'movie': batch[:, 1]}, batch[:, 2]

# Train the model
print("\n5. Training model...")
positive_samples_per_batch = 512
negative_ratio = 10
epochs = 25
steps_per_epoch = len(pairs) // positive_samples_per_batch

print(f"   Batch size: {positive_samples_per_batch * (1 + negative_ratio)}")
print(f"   Steps per epoch: {steps_per_epoch}")
print(f"   Total epochs: {epochs}")
print(f"   Training on ~{steps_per_epoch * epochs * positive_samples_per_batch:,} examples\n")

model.fit(
    generate_batches(pairs, positive_samples_per_batch, negative_ratio=negative_ratio),
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    verbose=2
)

# Save everything
print("\n6. Saving model and mappings...")
model.save('movie_embedding_model.keras')
print("✓ Model saved to 'movie_embedding_model.keras'")

with open('mappings.pkl', 'wb') as f:
    pickle.dump({
        'link_to_idx': link_to_idx,
        'movie_to_idx': movie_to_idx,
        'top_links': top_links,
        'all_movies': all_movies
    }, f)
print("✓ Mappings saved to 'mappings.pkl'")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nNext steps:")
print("  - Run inference.py to find similar movies")
print("  - Run svm.py to build a recommender system")
