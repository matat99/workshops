import pickle
import sys

# Get filename from command line argument
if len(sys.argv) < 2:
    print("Usage: python unpickle.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

# Unpickle and print
with open(filename, 'rb') as f:
    data = pickle.load(f)
    print(data)
