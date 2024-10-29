# INR-TPC-Compression

This repository contains the code to reproduce the results presented in the paper: **"Efficient Compression of Sparse Accelerator Data Using Implicit Neural Representations and Importance Sampling"**. The code implements methods to compress sparse accelerator data using various models, including FFNet, SIREN, and WIRE, combined with importance sampling techniques.

## Table of Contents
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Running the Experiments](#running-the-experiments)
- [Models](#models)
- [Results](#results)
- [License](#license)
- [Citation](#citation)

## Installation

### Prerequisites

- [Anaconda/Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your system.
- Python 3.8+ (The environment.yml file will handle this)

### Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/yourrepositoryname.git
    cd yourrepositoryname
    ```

2. **Create and activate the conda environment:**

    ```bash
    conda env create -f environment.yml
    conda activate INR_TPC
    ```

3. **Verify the installation:**

    Ensure that all dependencies are correctly installed by running:

    ```bash
    python -c "import torch; print(torch.__version__)"
    ```

## Directory Structure

The directory structure of this repository is as follows:

```plaintext
.
├── data/                   # Contains datasets and related files
├── models/                 # Contains model definitions
│   ├── ffnet.py            # FFNet model implementation
│   ├── siren.py            # SIREN model implementation
│   └── wire.py             # WIRE model implementation
├── dataload.py             # Data loading and preprocessing utilities
├── task_1.py               # Script for the first set of experiments
├── task_2.py               # Script for the second set of experiments
├── task_3.py               # Script for the third set of experiments
├── run.sh                  # Shell script to run all tasks sequentially
├── environment.yml         # Conda environment configuration
└── README.md               # Readme file
```

## Usage

### Data Preparation

Ensure that your data is stored in the `data/` directory. You may need to adjust paths in `dataload.py` or the experiment scripts (`task_1.py`, `task_2.py`, `task_3.py`) depending on your data structure.

### Running the Experiments

To reproduce the experiments presented in the paper, simply execute the provided shell script:

```bash
bash run.sh
```

This will sequentially run the experiments as defined in `task_1.py`, `task_2.py`, and `task_3.py`.

Alternatively, you can run each task individually:

```bash
python task_1.py
python task_2.py
python task_3.py
```

## Models

The following models are implemented in this repository:

- **FFNet**: Defined in `models/ffnet.py`
- **SIREN**: Defined in `models/siren.py`
- **WIRE**: Defined in `models/wire.py`

These models are designed to compress sparse accelerator data efficiently.

## Results

The results of the experiments will be stored in the `logs/` directory, which is automatically generated when running the scripts. Each task will create a subdirectory under `logs/` containing results, model checkpoints, and plots.

## License

This project is licensed under the MIT License.

## Citation

Coming soon.