"""
Shared model loader for all demo scripts.
Downloads once and caches locally for future use.
"""

import gensim.downloader as api
import os
from tqdm import tqdm

MODEL_NAME = 'word2vec-google-news-300'

def load_word2vec_model():
    """
    Load the Word2Vec model with progress indication.
    The model is cached locally after first download.
    """
    # Check if model is already downloaded
    cache_dir = api.base_dir
    model_path = os.path.join(cache_dir, MODEL_NAME)
    
    if os.path.exists(model_path):
        print(f"Loading cached model from: {cache_dir}")
        print("(Delete this folder to re-download)\n")
        model = api.load(MODEL_NAME)
        print("✓ Model loaded successfully!\n")
        return model
    
    print(f"Downloading {MODEL_NAME} model...")
    print(f"This is a one-time download (~1.6 GB)")
    print(f"Cache location: {cache_dir}\n")
    
    print("Downloading... (this may take several minutes)")
    model = api.load(MODEL_NAME, return_path=False)
    
    print("\n✓ Download complete! Model cached for future use.")
    print(f"✓ Model loaded successfully!\n")
    
    return model

if __name__ == "__main__":
    print("Pre-downloading Word2Vec model for all demo scripts")
    
    model = load_word2vec_model()
    
