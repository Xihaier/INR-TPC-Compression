#!/bin/bash

# Activate the conda environment
echo "Activating conda environment INR_TPC..."
source activate INR_TPC

# Check if the environment activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment. Exiting."
    exit 1
fi

# Run task_1.py
echo "Running task_1.py..."
python task_1.py

# Check if the script ran successfully
if [ $? -ne 0 ]; then
    echo "task_1.py failed. Exiting."
    exit 1
fi

# Run task_2.py
echo "Running task_2.py..."
python task_2.py

# Check if the script ran successfully
if [ $? -ne 0 ]; then
    echo "task_2.py failed. Exiting."
    exit 1
fi

# Run task_3.py
echo "Running task_3.py..."
python task_3.py

# Check if the script ran successfully
if [ $? -ne 0 ]; then
    echo "task_3.py failed. Exiting."
    exit 1
fi

echo "All tasks completed successfully."