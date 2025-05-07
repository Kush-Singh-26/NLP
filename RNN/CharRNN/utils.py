import io
import os
import unicodedata
import string
import glob
import torch
import random

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

# Converts a Unicode string to plain ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)  # Normalize string to decompose accents
        if unicodedata.category(c) != 'Mn'          # Filter out non-spacing marks (accents)
        and c in ALL_LETTERS                        # Keep only allowed characters
    )

# Loads training data from files
def load_data():
    category_lines = {}     # Dictionary mapping category -> list of names
    all_categories = []     # List of all categories (nationalities)
    
    # Returns a list of all matching file paths
    def find_files(path):
        return glob.glob(path)
    
    # Reads lines from a given file and converts them to ASCII
    def read_lines(filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    
    # Process each file in the data/names/ directory
    for filename in find_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]  # Extract category from filename
        all_categories.append(category)
        
        lines = read_lines(filename)        # Read and clean the names in the file
        category_lines[category] = lines    # Store them under the corresponding category
        
    return category_lines, all_categories

# Get index of a letter from the ALL_LETTERS string
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

# Converts a single letter to a one-hot encoded tensor of shape (1, N_LETTERS)
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

# Converts a name to a tensor of shape (line_length, 1, N_LETTERS), 1 = Batch size 
# Each letter is represented by a one-hot encoded vector
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor

# Selects a random (category, name) pair and returns the tensors
def random_training_example(category_lines, all_categories):
    
    # Helper function to randomly select an element from a list
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    # Randomly choose a category and a name from that category
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    
    # Create tensors: category as index tensor, name as one-hot encoded sequence
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    
    return category, line, category_tensor, line_tensor
