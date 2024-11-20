#!/bin/bash

# Define maximum number of threads to use from argument and default value 1
num_threads="${6:-1}"


# Check if the required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <type> <dataset> <sampling_reduction> <candidate_reduction> [MIN_SIZE_factor] [THREADS_NUM]"
    echo "For Sequential: $0 Sequential <dataset>"
    echo "For others: $0 <type> <dataset> <sampling_reduction> <candidate_reduction> [MIN_SIZE_factor] [THREADS_NUM]"
    exit 1
fi

# Take the mandatory arguments
type="$1"        # Type of the program (OpenMP, Sequential, OpenCilk, pthreads)
dataset="$2"     # Dataset to run the program on (e.g., "sift", "mnist", "fashion-mnist")
sampling_reduction="$3"  # Mandatory sampling reduction factor
candidate_reduction="$4" # Mandatory candidate reduction factor

# Default value for optional argument MIN_SIZE_factor
minsize_factor="${5:-1}" # Default minsize_factor = 1

# Validate arguments based on the type
if [ "$type" == "Sequential" ]; then
    # Sequential requires only type and dataset
    if [ $# -gt 2 ]; then
        echo "Too many arguments for Sequential. Usage: $0 Sequential <dataset>"
        exit 1
    fi
    sampling_reduction=1
    candidate_reduction=1
    minsize_factor=1
else
    # Other types require at least dataset, sampling_reduction, and candidate_reduction
    if [ $# -lt 4 ]; then
        echo "Insufficient arguments for $type. Usage: $0 <type> <dataset> <sampling_reduction> <candidate_reduction> [MIN_SIZE_factor] [THREADS_NUM]"
        exit 1
    fi
fi

# Define directories
code_dir="./../Code"
build_dir="./../Code/build"


# Move to the code directory to compile the program
cd "${code_dir}" || { echo "Failed to change directory to ${code_dir}. Exiting."; exit 1; }

# Compile the specific variant
make "$type"

# Verify if the build was successful
if [ ! -f "${build_dir}/${type}" ]; then
    echo "Build failed for ${type}. Exiting."
    exit 1
fi

# Move back to the original directory after compilation
cd - > /dev/null 2>&1 || exit 1

# Create a directory for the output results if it doesn't exist
results_dir="./${type}/$dataset/"
mkdir -p "$results_dir"
mkdir -p "${results_dir}Analytic"
    mkdir -p "${results_dir}Statistics"

# Define output file names based on parameters
    output_file="${results_dir}Analytic/results_${dataset}_${sampling_reduction}_${candidate_reduction}_${minsize_factor}.txt"
    statistics_output_file="${results_dir}Statistics/statistics_${dataset}_${sampling_reduction}_${candidate_reduction}_${minsize_factor}.txt"

# Clear the approximate output file before starting
> "$output_file"

echo -e "Running ${type} on ${dataset} dataset with sampling_reduction=${sampling_reduction}, candidate_reduction=${candidate_reduction}, minsize_factor=${minsize_factor}\n"

# Set the number of workers dynamically depending on the program type
if [ "$type" == "OpenCilk" ]; then
    # Set the CILK_NWORKERS environment variable and run OpenCilk
    export CILK_NWORKERS=$num_threads
    "${build_dir}/${type}" "${dataset}" "${sampling_reduction}" "${candidate_reduction}" "${minsize_factor}" >> "$output_file"
elif [ "$type" == "OpenMP" ]; then
    # Set the OMP_NUM_THREADS environment variable and run OpenMP
    export OMP_NUM_THREADS=$num_threads
    "${build_dir}/${type}" "${dataset}" "${sampling_reduction}" "${candidate_reduction}" "${minsize_factor}" >> "$output_file"
else
    "${build_dir}/${type}" "${dataset}" "${sampling_reduction}" "${candidate_reduction}" "${minsize_factor}" >> "$output_file"
fi

# Calculate statistics 
# echo -e "${type} results completed, now calculating statistics...\n"
# python3 calculate_recall.py "$output_file" "./Sequential/$dataset/Analytic/results_${dataset}_1_1_1.txt" "$statistics_output_file"
# echo -e "Statistics calculation completed."


# Print summary of results
# echo -e "\nAll runs and calculations are complete. Results saved to:"
# echo -e "Results file: $output_file"
# if [ "$type" != "Sequential" ]; then
#     echo -e "Statistics file: $statistics_output_file"
# fi
