"""
Find words most similar to a given word in the embedding space.
This demonstrates semantic clustering - similar words have similar vectors.
"""

from load_model import load_word2vec_model

# Load the pre-trained Word2Vec model from cache
model = load_word2vec_model()

def find_similar(word, topn=5):
    """Find the most similar words to the given word."""
    if word not in model:
        print(f"'{word}' not found in vocabulary")
        return
    
    print(f"Words most similar to '{word}':")
    similar = model.most_similar(word, topn=topn)
    for similar_word, score in similar:
        print(f"  {similar_word}: {score:.3f}")
    print()

# Try different words - modify these to explore!
find_similar('espresso', topn=5)
find_similar('python', topn=5)
find_similar('king', topn=5)

