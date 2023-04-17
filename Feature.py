import os
import numpy as np
import cv2
from LMTRP import LMTRP_process
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cdist

def extract_features(image_path):
    image = cv2.imread(image_path)
    feature = LMTRP_process(image)          
    return feature

# Load gallery vectors
X = np.load("features.npy")
print(X.shape)
gallery_vectors = X.reshape(X.shape[0], -1)

# Extract features of query image
query_feature = extract_features("./random/quan.bmp")
query_vector = query_feature.reshape(1, -1)

# Set a threshold for the minimum distance
threshold = 4.2

# Compute distances between query vector and gallery vectors
distances = cdist(query_vector, gallery_vectors, 'euclidean')

# Find the index of the closest gallery vector
min_index = np.argmin(distances)
print(distances)
print(distances[0][min_index])
# Check if the minimum distance is less than the threshold
if distances[0][min_index] < threshold:
    print("Vector is in the dataset.")
else:
    print("Vector is not in the dataset.")
