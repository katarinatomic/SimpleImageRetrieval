# +
import os
import numpy as np
import cv2
from typing import Sequence

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

class Vectorizer:
    def __init__(self, vectorized_images=None,debug=False):
        """
        Initialize the Vectorizer.
        
        Args:
            vectorized_images: Numpy array of vectorized images.
            debug: Flag to enable debug mode.
        """
        self.debug = debug
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        
    def resize_and_pad(self, image, target_shape=(224, 224, 3)):
        """Resizes and pads image while preserving aspect ratio.

        1.Keras preprocess: The images are converted from RGB to BGR,
            then each color channel is zero-centered with respect to the 
            ImageNet dataset, without scaling. 
        2.Resize and pad image: Resize the longer side to target size.
            Pad the shorter side to a square.
            
        Arguments:
            image: Input image as a numpy array.
            target_shape: The target shape for the image. 

        Returns:
            Resized and padded image as a numpy array.
        """
        image_shape = image.shape[:2]
        ratio = target_shape[0] / min(image_shape)
        if ratio * max(image_shape) > target_shape[0]:
            ratio = target_shape[0] / max(image_shape)
        new_shape = tuple([int(x*ratio) for x in image_shape])
        image_resized = cv2.resize(image, (new_shape[1], new_shape[0]))

        pad_height = max(target_shape[0] - image_resized.shape[0], 0)
        pad_width = max(target_shape[1] - image_resized.shape[1], 0)
        top, bottom = pad_height//2, pad_height-(pad_height//2)
        left, right = pad_width//2, pad_width-(pad_width//2)

        padded_image = np.pad(image_resized, ((top, bottom), (left, right), (0, 0)), mode='constant')
        return padded_image
    
    def normalize(self, image):
        """
        Normalizes image to range [0,1].
        """
        image = image.astype(np.float32)
        image = (image-image.min())/(image.max()-image.min())
        return image

    def preprocess_image(self, image):
        """Preprocess the image with resizing and padding.

        Arguments:
            image: Input image as a numpy array.
        Returns:
            Preprocessed image as a numpy array.
        """
        keras_preprocess = preprocess_input(image)
        image = self.resize_and_pad(keras_preprocess)
        #image = self.normalize(image)
        return image
    
    def transform(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Transform list of images into numpy vectors of image features.
        Args:
            images: A sequence of raw images.
        Returns:
            Vectorized images as numpy array of (N, D) shape where
            N is the number of images and D is the feature vector size.
        """
        preprocessed_images = [self.preprocess_image(img) for img in images]
        preprocessed_images = np.array(preprocessed_images)
        vectorized_images = self.model.predict(preprocessed_images)
        
        return np.array(vectorized_images)
    
    def save(self, array, filename='vectorized_images.npy'):
        """Saves vectorized images to file.
        Arguments:
           array: A numpy array of vectorized images.
           filename: Name of the file to save the array.
        """
        np.save(filename, array)
        
    def load(self, filename):
        """Loads presaved vectorized images from folder.
        Arguments:
           filename: Path to .npy file.
        Returns:
           array: A numpy array of vectorized images.
        
        """
        return np.load(filename)
# -


