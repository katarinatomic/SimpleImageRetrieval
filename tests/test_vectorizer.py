import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import unittest
import numpy as np
import sys
sys.path.append('../')
from lib.vectorizer import Vectorizer

class VectorizerTestCase(unittest.TestCase):
    def test_transform(self):
        # Test data
        #image_vectors = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        image_vectors = np.expand_dims(np.zeros([100,100,3],dtype=np.uint8),axis=0)
        # Create a Vectorizer instance
        vectorizer = Vectorizer()
        
        # Perform transformation
        transformed_vectors = vectorizer.transform(image_vectors)

        # Assert the transformed vectors are numpy arrays
        self.assertIsInstance(transforsmed_vectors, np.ndarray)

        # Assert the values of the transformed vectors
        #expected_values = np.array([])
        #np.testing.assert_allclose(transformed_vectors, expected_values, rtol=1e-5, atol=0)
    
    def test_save_and_load(self):
        # Test data
        vectorized_images = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        
        # Create a Vectorizer instance
        vectorizer = Vectorizer(vectorized_images=vectorized_images)
        
        # Save the vectorized images
        vectorizer.save(vectorized_images)
        
        # Load the saved vectorized images
        loaded_images = np.load('vectorized_images.npy')
        
        # Assert the loaded images are numpy arrays
        self.assertIsInstance(loaded_images, np.ndarray)
        
        # Assert the shape of the loaded images
        self.assertEqual(loaded_images.shape, vectorized_images.shape)
        
        # Assert the values of the loaded images
        np.testing.assert_allclose(loaded_images, vectorized_images, rtol=1e-5, atol=0)
        
        # Clean up the saved file
        os.remove('vectorized_images.npy')

if __name__ == '__main__':
    unittest.main()
