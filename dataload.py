import os
import random
import numpy as np


def load_dataset_info(base_path="data"):
    """
    Loads and validates dataset file paths from the specified base path.

    The function reads the file names from 'train.txt' and 'test.txt', checks if they are valid 
    `.npy` files, and returns a list of full file paths.

    Args:
        base_path (str): The base directory path where the dataset and file lists are stored.

    Returns:
        list: A list of valid `.npy` file paths.
    """
    # Load file names from train.txt and test.txt
    train_file_path = os.path.join(base_path, "train.txt")
    test_file_path = os.path.join(base_path, "test.txt")

    # Read and strip file paths from both train and test files
    train_files = [line.strip() for line in open(train_file_path)]
    test_files = [line.strip() for line in open(test_file_path)]

    # Combine train and test file paths and filter only valid `.npy` files
    all_files = [
        os.path.join(base_path, file_path) 
        for file_path in train_files + test_files 
        if os.path.isfile(os.path.join(base_path, file_path)) and file_path.endswith('.npy')
    ]

    # Log the total number of valid files found
    print(f"Total files found: {len(all_files)}")
    return all_files


def load_random_data(num_samples=50):
    """
    Loads a specified number of random data samples from the dataset.

    This function selects a random subset of files from the dataset, loads the `.npy` files, 
    and converts the data to `int16` format.

    Args:
        num_samples (int): The number of random samples to load. Defaults to 50.

    Returns:
        tuple: A tuple containing:
            - data_samples (list): A list of numpy arrays with the loaded data.
            - selected_files (list): A list of the file paths corresponding to the loaded data.
    """
    # Load all available dataset files
    all_files = load_dataset_info()

    # Ensure the requested number of samples does not exceed the available files
    num_samples = min(num_samples, len(all_files))

    # Randomly select a subset of files
    selected_files = random.sample(all_files, num_samples)

    # Load data from the selected files and convert to 'int16'
    data_samples = [np.load(file).astype(np.int16) for file in selected_files]

    return data_samples, selected_files


def load_ordered_data(base_path="data", num_samples=50):
    """
    Loads the first 50 `.npy` data samples from the dataset specified by the base path.

    This function reads file paths from 'train.txt' and 'test.txt', validates the files, and 
    loads the first 50 valid `.npy` files.

    Args:
        base_path (str): The base directory path where the dataset and file lists are stored. 
                         Defaults to "outer".

    Returns:
        tuple: A tuple containing:
            - data_samples (list): A list of numpy arrays with the loaded data.
            - all_files (list): A list of the file paths corresponding to the loaded data.
    """
    # Load file names from train.txt and test.txt
    train_file_path = os.path.join(base_path, "train.txt")
    test_file_path = os.path.join(base_path, "test.txt")

    # Read and strip file paths from both train and test files
    train_files = [line.strip() for line in open(train_file_path)]
    test_files = [line.strip() for line in open(test_file_path)]

    # Initialize a list to store valid file paths
    all_files = []

    # Iterate over combined train and test files, collecting up to 50 valid `.npy` files
    for file_path in train_files + test_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.isfile(full_path) and full_path.endswith('.npy'):
            all_files.append(full_path)
            if len(all_files) == num_samples:
                break

    # Load data from the selected files and convert to 'int16'
    data_samples = [np.load(file).astype(np.int16) for file in all_files]

    return data_samples, all_files


def transform_data(data):
    """
    Transforms the input data by applying a logarithmic transformation to elements greater than 64.
    
    The function creates a copy of the input data as a float array, applies the transformation 
    to elements greater than 64, and returns both the original and transformed arrays.

    Args:
        data (np.ndarray): The input array to be transformed.

    Returns:
        tuple: A tuple containing:
            - result (np.ndarray): A float copy of the original data.
            - transformed (np.ndarray): The transformed data where values > 64 are logarithmically scaled.
    """
    # Create a copy of the input array as float
    result = data.astype(float)
    
    # Create a mask for values greater than 64
    valid_mask = result > 64
    
    # Apply the logarithmic transformation only to valid elements
    transformed = result.copy()
    transformed[valid_mask] = np.log(transformed[valid_mask] - 64) / 6
    
    return result, transformed


def reverse_transform(data):
    """
    Reverses the transformation applied by `transform_data`, converting the data back to its original scale.
    
    The function applies the exponential transformation, adds 64, and sets any values below 67 to 0.
    The result is then cast to an integer array.

    Args:
        data (np.ndarray): The transformed data to be reversed.

    Returns:
        np.ndarray: The reversed data as an integer array.
    """
    # Apply the reverse transformation
    result = np.exp(data * 6) + 64
    
    # Set values below 67 to 0
    result[result < 67] = 0
    
    # Convert the result to an integer array
    result = result.astype(int)
    
    return result