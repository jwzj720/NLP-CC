import numpy as np
from pathlib import Path
from glove import load_glove_vectors
# Paths to the files
input_file = Path('data/glove.6B.50d.txt')
output_file = 'data/glove.6B.50d.npy'

words, vectors = load_glove_vectors(Path(output_file))


print(f"Number of words in original file: {len(words)}")
print(f"Shape of the loaded vectors array: {vectors.shape}")

print("\nFirst 5 words and their vectors from the original file:")
for i in range(5):
    print(f"{words[i]}: {vectors[i][1]}")

print_word = 'dog'
print(f"\nIndex of the word '{print_word}': {words.index(print_word)}")
print(f"Vector of the word '{print_word}': {vectors[words.index(print_word)]}")