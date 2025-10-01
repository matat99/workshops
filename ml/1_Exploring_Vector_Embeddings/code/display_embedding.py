import sys
from load_model import load_word2vec_model

if len(sys.argv) != 2:
    print("Usage: python show_vector.py <word>")
    sys.exit(1)

word = sys.argv[1]
model = load_word2vec_model()

if word not in model:
    print(f"'{word}' not found in vocabulary")
    sys.exit(1)

vector = model[word]

print(f"\nVector for '{word}' ({len(vector)} dimensions):\n")
print(vector)
