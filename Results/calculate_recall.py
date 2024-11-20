import sys

def load_results(filename):
    """Load k-NN results from a file and parse into a dictionary format."""
    results = {}
    execution_time = None
    dataset_size = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Query"):
                try:
                    # Parse the line in the format: "Query <q>, Neighbor <l>: <idx>"
                    parts = line.split(", ")
                    query_num = int(parts[0].split()[1])  # Extract query number
                    neighbor_info = parts[1].split(": ")
                    neighbor_idx = int(neighbor_info[1])  # Extract neighbor index
                    
                    # Initialize the query's neighbor set if it doesn't exist
                    if query_num not in results:
                        results[query_num] = set()
                    results[query_num].add(neighbor_idx)
                except (IndexError, ValueError) as e:
                    print(f"Error parsing line: {line} - {e}")
                    continue
            elif line.startswith("Execution time"):
                try:
                    # Parse the execution time line: "Execution time : <value> second"
                    execution_time = float(line.split(":")[1].split()[0])  # Extract execution time
                except (IndexError, ValueError) as e:
                    print(f"Error parsing execution time: {line} - {e}")
                    continue
            elif line.startswith("Size of dataset"):
                try:
                    # Parse the dataset size line: "Size of dataset: <value>"
                    dataset_size = int(line.split(":")[1])  # Extract dataset size
                except (IndexError, ValueError) as e:
                    print(f"Error parsing dataset size: {line} - {e}")
                    continue

    return results, execution_time, dataset_size

def calculate_recall(approx_results, brute_force_results):
    """Calculate recall as the ratio of correct neighbors in the approximate solution."""
    total_correct = 0
    total_neighbors = 0

    for query, brute_neighbors in brute_force_results.items():
        approx_neighbors = approx_results.get(query, set())
        correct_neighbors = len(brute_neighbors.intersection(approx_neighbors))
        
        total_correct += correct_neighbors
        total_neighbors += len(brute_neighbors)

    recall = total_correct / total_neighbors if total_neighbors > 0 else 0
    return recall

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python calculate_recall.py <approx_file> <brute_force_file> <output_file>")
        sys.exit(1)

    approx_file = sys.argv[1]
    brute_force_file = sys.argv[2]
    output_file = sys.argv[3]

    # Load results from both files
    approx_results, execution_time, dataset_size = load_results(approx_file)
    brute_force_results, BFexecution_time, _ = load_results(brute_force_file)

    # Calculate recall
    recall = calculate_recall(approx_results, brute_force_results)

    # Write results to output file
    with open(output_file, 'w') as f:
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Size of dataset: {dataset_size}\n")
        f.write(f"Queries per second: {dataset_size / execution_time:.4f}\n")
        f.write(f"Execution time (approximate): {execution_time:.4f} s\n")
        f.write(f"Execution time (brute force): {BFexecution_time:.4f} s\n")
        f.write(f"Speedup: {BFexecution_time / execution_time:.4f}\n")

    # Print results to console
    # print(f"Recall: {recall:.4f}")
    # print(f"Size of dataset: {dataset_size}")
    # print(f"Queries per second: {dataset_size / execution_time:.4f}")
    # print(f"Execution time (approximate): {execution_time:.4f} s")
    # print(f"Execution time (sequential): {BFexecution_time:.4f} s")
    # print(f"Speedup: {BFexecution_time / execution_time:.4f}")