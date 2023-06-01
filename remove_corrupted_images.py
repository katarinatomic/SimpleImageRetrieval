import os
import argparse
import glob
from lib.utils import read_and_check_image

def remove_corrupted_images(dataset_path):
    print('Scanning for corrupted images...')
    corrupted = []
    for path in glob.glob(dataset_path+'/*'):
            is_corrupted, _ = read_and_check_image(path)
            if is_corrupted:
                corrupted.append(path)
                os.remove(path)
    return corrupted

# +
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--dataset_path', type=str, default='simple_image_retrieval_dataset/image-db/')

#     sys.argv = ['remove_corrupted_images.py',
#                 '-p', '/path/to/image_dataset',
#                 ]
    args = parser.parse_args()
    
    corrupted = remove_corrupted_images(args.dataset_path)
    print('Removed the following images:',corrupted)
