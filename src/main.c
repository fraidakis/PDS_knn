#include <stdio.h>
#include "knn.h"

int main(int argc, char *argv[])
{
    int num_query;     // Number of queries
    const int k = 100; // Number of nearest neighbors to find

    const char *path = "./../Datasets/";                        // Path to datasets
    char filename[128];                                                                                    // Filename for dataset to import
    strcpy(filename, "./../Datasets/mnist-784-euclidean.hdf5"); // Default dataset

    if(argc == 5)
        update_file_name(filename, path, argv[1]); // Update filename based on command line arguments

    Matrix Q;                              // Query matrix
    import_matrix_from_file(filename, &Q); // Import query data from file

    Q.rows /= 1;                // Reduce the number of queries for testing purposes (eg big datasets)
    num_query = Q.rows;           // Get number of queries in the dataset
    int CORPUS_CHUNK_SIZE = 1000; // Set maximum chunk size

    // Compute dynamic parameters based on number of queries (default values)
    int MIN_SIZE = compute_min_size(num_query) / 1; // Set minimum size for brute-force search
    int sampling_reduction = 20;                    // Set sampling reduction factor
    int candidate_reduction = 100;                  // Set candidate reduction factor

    // Update parameters based on command line arguments
    if (argc == 5)
        update_parameters(argv, num_query, &sampling_reduction, &candidate_reduction, &MIN_SIZE);

    int height = (int)log2f(num_query / (MIN_SIZE - 1)); // Set height of the recursion tree

    int *idx = (int *)malloc(num_query * k * sizeof(int)); // Indices for k-NN search
    float *dst = malloc(num_query * k * sizeof(float));    // Distances for k-NN search
    float *squared_norms = malloc(Q.rows * sizeof(float)); // Squared norms for query data

    double start, end; // Variables to store execution time

    start = get_time_in_seconds(); // Start timer

    compute_norms(&Q, squared_norms); // Precompute squared norms of the query data

    if (strstr(argv[0], "Sequential"))
        knnsearch_chunked(&Q, &Q, k, idx, dst, CORPUS_CHUNK_SIZE, squared_norms, 0); // Perform k-NN search
    else
    {
        knnsearch_recursive(&Q, idx, dst, k, MIN_SIZE, sampling_reduction, candidate_reduction,
                            CORPUS_CHUNK_SIZE, height, 0, 0, squared_norms); // Perform k-NN search
    }

    end = get_time_in_seconds(); // Stop timer

    double elapsed_time = end - start; // Calculate elapsed time

    // Print results
    for (int q = 0; q < Q.rows; q++)
    {
        for (int l = 0; l < k; l++)
        {
            printf("Query %d, Neighbor %d: %d\n", q, l, idx[q * k + l]);
        }
        printf("\n");
    }

    printf("\nExecution time : %f second\n", elapsed_time);
    printf("Size of dataset: %d\n", Q.rows);

    // Free memory
    free(idx);
    free(dst);
    free(squared_norms);
    free(Q.data);

    return 0;
}
