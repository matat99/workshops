"""
t-SNE: Visualizing high-dimensional embeddings in 2D
This shows how similar words cluster together in the embedding space.

t-SNE (t-distributed Stochastic Neighbor Embedding) projects 300-dimensional
vectors down to 2D while preserving neighborhood relationships.
"""

from load_model import load_word2vec_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Configure matplotlib for better visualization
plt.rcParams['figure.figsize'] = (10, 10)

# Load the pre-trained Word2Vec model from cache
model = load_word2vec_model()

# Define categories of words to visualize
beverages = ['espresso', 'beer', 'vodka', 'wine', 'cola', 'tea']
countries = ['Italy', 'Germany', 'Russia', 'France', 'USA', 'India']
sports = ['soccer', 'handball', 'hockey', 'cycling', 'basketball', 'cricket']

# Combine all items
items = beverages + countries + sports

# Get vectors for items that exist in the model
print(f"Extracting vectors for {len(items)} words...")
item_vectors = [(item, model[item]) for item in items if item in model]
print(f"Found {len(item_vectors)} words in model vocabulary\n")

# Prepare vectors for t-SNE
vectors = np.asarray([x[1] for x in item_vectors])

# Normalize vectors (optional but often helpful)
lengths = np.linalg.norm(vectors, axis=1)
norm_vectors = (vectors.T / lengths).T

# Apply t-SNE to reduce from 300D to 2D
print("Running t-SNE (this may take a moment)...")
print("Perplexity=10, this controls how many neighbors each point considers")
tsne = TSNE(
    n_components=2,      # Output dimensions
    perplexity=10,       # Balance between local and global structure
    verbose=1,           # Show progress
    random_state=42      # For reproducibility
).fit_transform(norm_vectors)

# Extract x and y coordinates
x = tsne[:, 0]
y = tsne[:, 1]

# Create the visualization
print("Creating visualization...\n")
fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.6, s=100)

# Label each point
for item, x1, y1 in zip(item_vectors, x, y):
    ax.annotate(item[0], (x1, y1), size=14)

ax.set_title('t-SNE Visualization of Word Embeddings', size=16)
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')

plt.tight_layout()
plt.show()
