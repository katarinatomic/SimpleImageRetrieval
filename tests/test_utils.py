import unittest
import numpy as np
from pathlib import Path
import os
import sys
sys.path.append('../')
from lib.utils import read_and_check_image, load_images, load_image_names

class TestUtils(unittest.TestCase):
    def test_read_and_check_image(self):
        # Test a valid image file
        is_corrupted, image = read_and_check_image("valid_image.jpg")
        self.assertFalse(is_corrupted)
        self.assertIsInstance(image, np.ndarray)
        
        # Test a corrupted image file
        is_corrupted, image = read_and_check_image("corrupted_image.jpg")
        self.assertTrue(is_corrupted)
        self.assertIsNone(image)
