# Execution Guide

## Overview
---
Script location: `./Results/run.bash` 

It is used to compile and run different implementations (`Sequential`, `OpenMP`, `OpenCilk`, and `Pthreads`) on specified datasets with configurable parameters.

---
## Prerequisites
Before running the script, make sure the following are installed:
- `make`
- Required dependencies for compiling `OpenMP`, `OpenCilk`, and `Pthreads` implementations

## Running the Script

### Script Usage
To run the script, you need to specify the type of implementation (`Sequential`, `OpenMP`, `OpenCilk`, `Pthreads`) along with the dataset and configuration parameters.


## How to Execute

### Basic Command
```bash
bash run_script.sh <type> <dataset> <sampling_reduction> <candidate_reduction> [MIN_SIZE_factor] [THREADS_NUM]
```

- `<type>`: The implementation type (`Sequential`, `OpenMP`, `OpenCilk`, `Pthreads`).
- `<dataset>`: The dataset to run the program on ("mnist", "fashion-mnist", "sift").
- `<sampling_reduction>`: The sampling reduction factor (integer).
- `<candidate_reduction>`: The candidate reduction factor (integer).
- `[MIN_SIZE_factor]` (optional): To divide MIN_SIZE (default is `1`).
- `[THREADS_NUM]` (optional): The number of threads to use (default is `1`).
    
    - **Note**: For `Pthreads`, the number of threads must be manually specified in the code by changing the `MAX_THREADS` define in `./src/knn_Pthreads.c`.


### Example Commands

1. **Run Sequential Implementation:**

   ```bash
   bash run.bash Sequential mnist
   ```
   
2. **Run OpenCilk Implementation with 4 threads:**

   This command will run the `OpenCilk` implementation on the `mnist` dataset with a sampling reduction of `32`, candidate reduction of `16`, and a minimum size factor of `2`. It will use `4` threads.
   
   ```bash
   bash run.bash OpenCilk mnist 25 200 1 12
   ```

## Output

After the script has completed running, the results will be saved in the following directory structure:

- **Analytic Results**:
  
  - `./<type>/<dataset>/Analytic/results_<dataset>_<sampling_reduction>_<candidate_reduction>_<minsize_factor>.txt`

- **Statistics** (only for `OpenCilk`, `OpenMP`, and `Pthreads`):
  
  - `./<type>/<dataset>/Statistics/statistics_<dataset>_<sampling_reduction>_<candidate_reduction>_<minsize_factor>.txt`

The directory structure will be created if it does not already exist.
