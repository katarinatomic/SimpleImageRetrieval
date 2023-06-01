import numpy as np
import sys
sys.path.append('../')
from lib.cosine_similarity import cosine_similarity

def test_cosine_similarity():
    query_image_vector = np.array([1, 2, 3])
    image_vectors = np.array([[4, 5, 6], [7, 8, 9], [30, 25, 12], [1, 2, 4]],)
    expected_similarities = np.array([0.99146, 0.974632, 0.959412, 0.758867])
    expected_indices = np.array([3,0,1,2])

    # Call the cosine_similarity function
    top_similarities, top_indices = cosine_similarity(query_image_vector, image_vectors, n=4)

    # Assert that the returned values match the expected values
    np.testing.assert_allclose(top_similarities, expected_similarities, rtol=1e-4)
    np.testing.assert_array_equal(top_indices, expected_indices)
