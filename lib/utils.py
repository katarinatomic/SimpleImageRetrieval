# +
import os
from pathlib import Path
import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from typing import List,Tuple
                
def read_and_check_image(image_path: str) -> Tuple[bool, np.ndarray]:
    """
    Check if image is corrupted.
    Arguments:
       image_path: A string path to image directory.
    Returns:
       bool: True is image is corrupted
       image: If image is not corrupted, None if is corrupted
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return True, None  # corrupted
        else:
            return False, img  # not corrupted
    except cv2.error:
        return True, None # corrupted

def load_images(root: Path, sufix: str = 'image-db/*') -> List[np.ndarray]:
    """
    Loading images. Dataset structure and names are assumed to be the same as downloaded.
    """
    images = []
    for filename in root.glob(sufix):
        is_corrupted, image = read_and_check_image(str(filename))
        if not is_corrupted:
            images.append(image)
    return images

def load_image_names(root: Path, sufix: str = 'image-db/*') -> List[str]:
    """
    Loading image filenames. Before using this method, use remove_corrupted_images().
    """
    return [filename for filename in root.glob(sufix)]
        
def debug_save_images(preprocessed_images: List[np.ndarray]) -> None:
    """Helper function for debugging. Loads a random
    sequence of images and saves them to debug/ folder.
    Arguments:
       preprocessed_images: A list of images extracted from the
        preprocessing function of vectorizer.
    """
    import os
    import random
    os.makedirs('./debug', exist_ok=True)
    
    def plot_images(image_sequence, save_path='debug/example_images.jpg'):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3,4, figsize=(15, 15))
        fig.tight_layout()
        for i, sample in enumerate(image_sequence):
            ax = plt.subplot(4, 3, i+1)
            ax.imshow(sample)
            ax.set_title(sample.shape)
            ax.axis('off')
        plt.savefig(save_path)
    
    random_int_sequence = [random.randint(0, len(preprocessed_images)) for i in range(12)]
    random_image_sequence = [preprocessed_images[index] for index in random_int_sequence]
    plot_images(random_image_sequence)
    
def plot_collage(top_similarities: List[float],
    top_indices: List[int],
    image_paths: List[str],
    save_filename: str ='cosine_similarity.jpg'
    ) -> None: 
    """
    Creates a collage of the top N most similar images from the data set.
    Plots similarity score and image name. The top N images are
    sorted by similarity score in descending order.
    """
    os.makedirs('./results', exist_ok=True)
    n = len(top_indices)
    fig, axs = plt.subplots(n,1, figsize=(15, 5))
    fig.tight_layout()

    for i, (score,index) in enumerate(zip(top_similarities, top_indices)):
        ax = plt.subplot(1,n, i+1)
        image = cv2.imread(str(image_paths[index]))
        ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        if i==0:
            ax.set_title('Query image')
        else:
            ax.set_title(f'{image_paths[index].name}\n score: {score:.3f}')
        ax.axis('off')
    plt.savefig(os.path.join('./results',save_filename), transparent = True, bbox_inches = 'tight', pad_inches = 0)
    print(f'Saved similarity collage to ./results/{save_filename}.')
