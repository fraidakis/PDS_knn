# Parallel and Distributed Systems - Exercise 1
This repository contains an implementation of the approximate k-Nearest Neighbors (ANN) algorithm using multiple parallelization techniques: `Sequential`, `OpenMP`, `OpenCilk`, and `Pthreads`

## Overview

The project implements the **k-Nearest Neighbors (k-NN)** algorithm, which finds the closest data points to a given query point. This project aims to improve the k-NN algorithm's performance using parallelization strategies to handle large datasets efficiently. The project uses the **ANN (Approximate k-NN)** approach for faster processing at the cost of a small loss in accuracy (recall).


## Requirements
Before running the script, ensure that you have the following installed:

- **make**: For building the project.
- **OpenMP**: Install the OpenMP library for multi-threaded execution.
- **OpenCilk**: Install the OpenCilk runtime for parallel execution.
- **Pthreads**: Ensure that the Pthreads library is available for thread-based parallelization.
- **OpenBLAS**: For fast linear algebra computations, used in distance calculation.
- **Linux/Unix environment**: For running the bash scripts and performance tools.

## How to Run

To compile and execute different implementations on specified datasets, follow these steps.


Script location:

### Script Usage
The script used for running benchmarks and experiments is located at:

```bash
 `./Results/run.bash`
```

- **Note**: Before running parallel implementations for a dataset, you must first execute the `Sequential` implementation (need the results for recall and speedup calculations)

## How to Execute

### Basic Command
```bash
bash run_script.sh <type> <dataset> <sampling_reduction> <candidate_reduction> [MIN_SIZE_factor] [THREADS_NUM]
```

- `<type>`: The implementation type (`Sequential`, `OpenMP`, `OpenCilk`, `Pthreads`).
- `<dataset>`: The dataset to run the program on (`mnist`, `fashion-mnist`, `sift`).
- `<sampling_reduction>`: The sampling reduction factor (positive integer).
- `<candidate_reduction>`: The candidate reduction factor (positive integer).
- `[MIN_SIZE_factor]` (optional): To divide MIN_SIZE (default is `1`).
- `[THREADS_NUM]` (optional): The number of threads to use (default is `1`).
    
    - **Note**: For `Pthreads`, the number of threads must be manually specified in the code by changing the `MAX_THREADS` define in `./src/knn_Pthreads.c`.


### Example Commands

1. **Run Sequential Implementation:**

   ```bash
   bash run.bash Sequential mnist
   ```
   
2. **Run OpenCilk Implementation with 12 threads:**

   This command will run the `OpenCilk` implementation on the `mnist` dataset with a sampling reduction of `25`, candidate reduction of `200`, and a minimum size factor of `1`. It will use `12` threads.
   
   ```bash
   bash run.bash OpenCilk mnist 25 200 1 12
   ```

## Output

After the script has completed running, the results will be saved in the following directory structure:

- **Analytic Results**:
  
  - Location: `./<type>/<dataset>/Analytic/results_<dataset>_<sampling_reduction>_<candidate_reduction>_<minsize_factor>.txt`
  - Contains indices of k nearest neighbors for each query.
      

- **Statistics** 
  
  - Location: `./<type>/<dataset>/Statistics/statistics_<dataset>_<sampling_reduction>_<candidate_reduction>_<minsize_factor>.txt`
  - Summarizes performance metrics, including execution time, recall, and queries per second.


The directory structure will be created if it does not already exist.
