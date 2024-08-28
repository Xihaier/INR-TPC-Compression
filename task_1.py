import torch
import time
import numpy as np
import dataload
import itertools
import pandas as pd
import logging

from pathlib import Path
from typing import List, Dict, Any, Type
from datetime import datetime
from models.siren import SIREN
from models.ffnet import FFNet
from models.wire import WIRE


def setup_logging(log_dir: str = "logs/task_1", log_file: str = "experiment.log") -> logging.Logger:
    """
    Set up logging to track the experiments.
    
    :param log_dir: Directory where logs will be saved.
    :param log_file: The name of the log file.
    :return: Configured logger object.
    """
    # Get current timestamp for directory naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the full log directory path
    log_path = Path(log_dir) / timestamp
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("ExperimentLogger")
    logger.setLevel(logging.INFO)
    
    # File handler for logging
    file_handler = logging.FileHandler(log_path / log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter for logging
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Adding handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Attach the log directory path as an attribute to the logger
    logger.log_dir = log_path

    return logger


def reshape_data(data: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Reshape the input data to the target shape by downsampling specific dimensions.
    """
    # Define the expected original shape
    original_shape = (192, 249, 16)
    
    # Validate the shape of the input data
    if data.shape != original_shape:
        raise ValueError(f"Input data must have shape {original_shape}, but got {data.shape}.")
    
    # If the target shape is the same as the original shape, return the data as-is
    if target_shape == original_shape:
        return data
        
    # Downsampling data along specified dimensions
    result = data.copy()
    if target_shape[0] == 96:
        result = result[::2, :, :]
    if target_shape[0] == 48:
        result = result[::4, :, :]
    if target_shape[1] == 125:
        result = result[:, ::2, :]
    if target_shape[1] == 63:
        result = result[:, ::4, :]
    if target_shape[2] == 8:
        result = result[:, :, ::2]
    
    return result


def evaluate_model(logger: logging.Logger, model, original_data: np.ndarray, transformed_data: np.ndarray, max_iters: int, grid_size: tuple):
    """
    Evaluate the performance of a model by training on transformed data and testing on original data.
    """
    start_time = time.time()

    # Reshape the transformed data to match the required grid size
    transformed_data = reshape_data(transformed_data, grid_size)
    
    # Train the model using the transformed data
    mse_transformed = model.train(transformed_data, total_steps=max_iters, summary_interval=2)
    training_time = time.time() - start_time
    
    # Test the model on the original data
    array_loader = model.create_loader(original_data)
    grid, array = next(iter(array_loader))
    
    # Prepare the grid and array for model prediction
    grid = grid.squeeze().to(model.device)
    array = array.squeeze().to(model.device)
    coords, values = grid.reshape(-1, 3), array.reshape(-1, 1)
    
    # Predict using the model on the original coordinates
    prediction_transformed = model.predict(coords)
    
    # Reshape the prediction to match the original data shape
    original_shape = array.cpu().numpy().shape
    predicted_array_transformed = prediction_transformed.reshape(original_shape).cpu().numpy()
    
    # Apply inverse transformation to bring predictions back to the original scale
    prediction_original = dataload.reverse_transform(predicted_array_transformed)

    # Calculate Mean Squared Error (MSE) on the original data scale
    mse_original = np.mean((original_data - prediction_original) ** 2)
    
    # Calculate the compression ratio achieved by the model
    compression_ratio = model.get_compression_ratio(original_data.size)
    
    logger.info(f"Model: {model.__class__.__name__}, MSE (original scale): {mse_original}, MSE (transformed scale): {mse_transformed}, Compression Ratio: {compression_ratio}, Training time: {training_time}")
    
    return mse_original, mse_transformed, compression_ratio, training_time, prediction_original


def evaluate_models(
    logger: logging.Logger,
    hyperparameters: Dict[str, Any], 
    original_samples: List[np.ndarray], 
    transformed_samples: List[np.ndarray], 
    file_names: List[str], 
    model_classes: List[Type]
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple models on given data samples using specified hyperparameters.
    """
    results = []
    
    for i, (original_data, transformed_data, file_name) in enumerate(zip(original_samples, transformed_samples, file_names)):
        logger.info(f"\nEvaluating on data sample {i + 1}: {file_name}")
        
        for ModelClass in model_classes:
            # Initialize the model with the provided hyperparameters
            model = ModelClass(hyperparameters)
            
            logger.info(f"\nEvaluating {ModelClass.__name__}")
            logger.info("Hyperparameters:")
            for key, value in hyperparameters.items():
                logger.info(f"  {key}: {value}")
            
            # Evaluate the model
            mse_original, mse_transformed, compression_ratio, training_time, prediction_original = evaluate_model(
                logger, 
                model, 
                original_data, 
                transformed_data, 
                hyperparameters["max_iters"], 
                hyperparameters["grid_size"]
            )
            
            # Save the prediction only for the first sample to avoid redundancy
            if i == 0:
                prediction_filename = f"{ModelClass.__name__}_sample{i + 1}_grid{hyperparameters['grid_size']}.npy"
                target_filename = f"tar_{ModelClass.__name__}_sample{i + 1}_grid{hyperparameters['grid_size']}.npy"
                prediction_path = logger.log_dir / prediction_filename
                target_path = logger.log_dir / target_filename
                np.save(prediction_path, prediction_original)
                np.save(target_path, original_data)
            
                # Store the results along with the prediction file path
                results.append({
                    'model': ModelClass.__name__,
                    'data_sample': i + 1,
                    'file_name': file_name,
                    'original_mse': mse_original,
                    'compression_ratio': compression_ratio,
                    'transformed_mse': mse_transformed,
                    'training_time': training_time,
                    'prediction_path': str(prediction_path),
                    **hyperparameters
                })
            else:
                # Store the results without saving the prediction file
                results.append({
                    'model': ModelClass.__name__,
                    'data_sample': i + 1,
                    'file_name': file_name,
                    'original_mse': mse_original,
                    'compression_ratio': compression_ratio,
                    'transformed_mse': mse_transformed,
                    'training_time': training_time,
                    **hyperparameters
                })                
    
    return results


def run_experiments(base_grid: Dict[str, List], model_grids: Dict[Type, Dict[str, List]]) -> pd.DataFrame:
    """
    Run experiments with different models and hyperparameters, logging the results.
    """
    logger = setup_logging()
    logger.info("Starting experiment run...")
    
    all_results = []
    
    # Load the first five random data samples along with their file names
    original_samples, file_names = dataload.load_ordered_data(num_samples=50)

    # Apply log transformation to the data samples
    transformed_samples = [
        dataload.transform_data(sample)[1] for sample in original_samples
    ]

    # Iterate over each model class and its corresponding grid of hyperparameters
    for ModelClass, grid in model_grids.items():
        # Generate all combinations of hyperparameters
        hyperparameter_combinations = [
            dict(zip(grid.keys(), values)) 
            for values in itertools.product(*grid.values())
        ]
        
        # Evaluate the model for each combination of hyperparameters
        for hyperparameters in hyperparameter_combinations:
            results = evaluate_models(
                logger,
                hyperparameters, 
                original_samples, 
                transformed_samples, 
                file_names, 
                [ModelClass]
            )
            all_results.extend(results)
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results to a CSV file
    results_df.to_csv(f'{logger.log_dir}/performance_summary.csv', index=False)
    
    logger.info("Experiment run complete.")
    
    return results_df


if __name__ == "__main__":
    import os 
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

    # Define the base grid of hyperparameters common to all models
    base_grid = {
        'lr': [1e-3],
        'in_features': [3],
        'hidden_features': [128],
        'hidden_layers': [3],
        'out_features': [1],
        'device': ['cuda:0'],
        'max_iters': [3000],
        'grid_size': [(192, 249, 16), (96, 125, 16), (96, 125, 8), (48, 63, 16)]
    }

    # Define model-specific grids with additional hyperparameters
    model_grids = {
        SIREN: {
            **base_grid, 
            'outermost_linear': [True],
            'first_omega_zero': [105],
            'hidden_omega_zero': [25]
        },
        WIRE: {
            **base_grid, 
            'a': [50], 
            's': [10]
        },
        FFNet: {
            **base_grid, 
            'input_scale': [300.0], 
            'weight_scale': [1.0]
        }
    }

    # Execute the experiment run with the defined hyperparameter grids
    results = run_experiments(base_grid, model_grids)

    # Display the final results
    print("\nFinal Results:")
    print(results)