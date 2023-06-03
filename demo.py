import cv2
from pathlib import Path
from main import main

def demo():
    try:
        query = cv2.imread(str(query_path))
        if query is None:
            print("Failed to load the query image.")
            return
        if len(query.shape)!=3:
            print("Query image is not color.")
            return
        #main function
        main(query, query_path, n)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# run demo
n = 5
query_path = '../test_1/simple_image_retrieval_dataset/test-cases/cat.jpg'
#dataset_root = '../simple_image_retrieval_dataset/'
demo()


