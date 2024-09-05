import numpy as np


# Function to extract all training times from a log file
def extract_training_times(log_file_path):
    training_times = []
    
    # Open the log file
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        
        # Iterate through each line starting at row 19 and step by 18
        for i in range(18, len(lines), 18):  
            line = lines[i]
            # Extract training time
            if "Training time:" in line:
                # Split line and extract the number after 'Training time:'
                training_time = float(line.split("Training time:")[1].strip())
                training_times.append(training_time)
    
    return training_times


def calculate_averages(arr, chunk_size=100):
    """
    Calculates the averages of the numbers in the array in chunks of specified size.

    Args:
        arr (numpy.ndarray): The input array of numbers.
        chunk_size (int, optional): The size of each chunk. Defaults to 20.

    Returns:
        numpy.ndarray: An array of averages for each chunk.
    """
    # Ensure the input array length is a multiple of the chunk size
    if len(arr) % chunk_size != 0:
        raise ValueError("Array length must be a multiple of the chunk size.")
    
    # Calculate the averages for each chunk
    averages = np.mean(arr.reshape(-1, chunk_size), axis=1)
    
    return averages


# Example usage
log_file_path = 'experiment.log'  # Replace with the actual path to your experiment.log file
training_times = extract_training_times(log_file_path)
training_times = calculate_averages(np.array(training_times))
print(training_times)
