# Simple Image Retrieval

This is a Python codebase for an Image Retrieval Engine that allows users to find similar images based on a query image. The codebase utilizes image vectorization and cosine similarity to calculate the similarity between images.

![](https://github.com/katarinatomic/SimpleImageRetrieval/blob/main/results/cat.jpg)
## Installation

1. Clone the repository:

    ```
    git clone https://github.com/katarinatomic/SimpleImageRetrieval.git
    ```

2. Install the required dependencies:

    ```pip install -r requirements.txt```


3. Prepare dataset by placing it in the `simple_image_retrieval_dataset/` directory.

## Usage

1. Remove corrupted images from the dataset:

    ```python remove_corrupted_images.py -p /path/to/image_dataset```

This step will scan the dataset for corrupted images and remove them. If you don't specify the dataset path, the default path is `simple_image_retrieval_dataset/image-db/`

2. Perform image search:

    ```python main.py --image <query-image-path> --top_n <num-results>```

Replace `<query-image-path>` with the path to your query image, and `<num-results>` with the number of similar images to retrieve.

## Dataset

If this repo is meant for you, you already have it :)

## Example

Here's an example of how to use the Image Search Engine:

1. Remove corrupted images:

    ```python remove_corrupted_images.py```

2. Perform image search:

    ```python main.py --image simple_image_retrieval_dataset/image-db/00000.jpg --top_n 5```

This command will search for the top 5 similar images to the query image `dataset/image-db/00000.jpg`.

3. A small demo.py script is added.
## Authors

- [@katarinatomic](https://github.com/katarinatomic)