# +
import numpy as np
import typing

def cosine_similarity(query_image_vector: np.ndarray, image_vectors: np.ndarray, n: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    # Compute the dot product between the query image vector and the image vectors
    dot_product = np.dot(query_image_vector, image_vectors.T)
    
    # Calculate the magnitudes of the query image vector and the image vectors
    query_dist = np.linalg.norm(query_image_vector)
    image_dist = np.linalg.norm(image_vectors, axis=1)
    
    # Calculate the cosine similarity using the dot product and magnitudes
    cosine_similarities = dot_product / (query_dist * image_dist)
    
    # Sort the cosine similarities in descending order and get the indices of the top n values
    top_indices = np.argsort(cosine_similarities)[::-1][:n]
    
    # Return the top n most similar images
    top_similarities = cosine_similarities[top_indices]
    
    return top_similarities, top_indices
