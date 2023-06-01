# +
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
from pathlib import Path
from typing import List,Tuple
import argparse

from lib.vectorizer import Vectorizer
from lib.search_engine import ImageSearchEngine
from lib.utils import load_image_names,load_images,plot_collage


def main(query: np.ndarray,
         query_path: str,
         n: int,
         dataset_path: str ='simple_image_retrieval_dataset/'
        ) -> List[Tuple[float, str]]:
    
    root = Path(dataset_path)
    image_paths = load_image_names(root)

    if os.path.isfile('vectorized_images.npy'):
        print('Vectorized images already exist in folder!')
        vectorized_images = np.load('vectorized_images.npy')
        vectorizer = Vectorizer(vectorized_images=vectorized_images)
    else:
        print('Generating new vectorized images...')
        image_vectors = load_images(root)
        vectorizer = Vectorizer()
        vectorized_images = vectorizer.transform(image_vectors)
        vectorizer.save(vectorized_images)
        print('Finished!')

    engine = ImageSearchEngine(vectorizer, vectorized_images, image_paths)
    top_similarities,top_indices = engine.most_similar(query, n)
    plot_collage(top_similarities, top_indices, image_paths, query_path.split('/')[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='simple_image_retrieval_dataset/')
    parser.add_argument('--top_n', default=5, help='Number of similar images to display.')

    args = parser.parse_args()
    query_image = cv2.imread(str(args.image))
    
    n = int(args.top_n)
    output = main(query_image, str(args.image), n)
