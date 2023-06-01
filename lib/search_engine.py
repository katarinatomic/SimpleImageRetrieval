# +
from lib.vectorizer import Vectorizer
from lib.cosine_similarity import cosine_similarity
from typing import List,Tuple
import numpy as np

class ImageSearchEngine:
    def __init__(self, vectorizer: Vectorizer, image_vectors: np.ndarray, image_paths: List[str]):
        self.vectorizer = vectorizer
        self.image_vectors = image_vectors
        self.image_paths = image_paths

    def most_similar(
        self,
        query: np.ndarray,
        n: int = 5
        ) -> List[Tuple[float, str]]:
        """Return top n most similar images from corpus.
            Input image should be cleaned and vectorized with fitted
            Vectorizer to get query image vector. After that, use
            the cosine_similarity function to get the top n most similar
            images from the data set.

        Args:
            query: The raw query image input from the user.
            n: The number of similar image names returned from the corpus.
        Returns:
            The list of top n most similar images from the corpus along
            with similarity scores. Note that returned str is image name.
        """
        query_vector = self.vectorizer.transform([query])
        top_similarities, top_indices  = cosine_similarity(query_vector[0], self.image_vectors, n+1)
        return top_similarities,top_indices
