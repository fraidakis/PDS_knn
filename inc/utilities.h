#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdlib.h>
#include <string.h>     
#include <stdalign.h> 
#include <sys/time.h>
#include <cblas.h>     
#include <hdf5.h>

/**
 * Matrix structure representing a dense matrix with data stored as a 1D array
 * of floats, where each row is stored contiguously.
 */
typedef struct
{
 float *data; /**< Pointer to the matrix data, ensuring 32-byte alignment */     
    int rows;    /**< Number of rows/points in the matrix */
    int cols;    /**< Number of columns/dimensions in the matrix */
} Matrix;

/**
 * Returns the current time in seconds as a double. Useful for measuring
 * execution time of functions or code segments.
 *
 * @return Current time in seconds.
 */
double get_time_in_seconds(void);

/**
 * Updates the filename based on command line arguments.
 *
 * @param filename Pointer to the filename to update.
 * @param path Path to the datasets.
 * @param dataset Name of the dataset to import.
 */
void update_file_name(char *filename, const char *path, char *dataset);

/**
 * Updates the parameters based on command line arguments.
 *
 * @param argv Command line arguments.
 * @param num_query Number of query points.
 * @param sampling_reduction Pointer to the sampling reduction factor.
 * @param candidate_reduction Pointer to the candidate reduction factor.
 * @param MIN_SIZE Pointer to the minimum size for brute-force search.
 */
void update_parameters(char* argv[], int num_query, int *sampling_reduction, int *candidate_reduction, int *MIN_SIZE);

/**
 * Loads a dense matrix from an HDF5 file into the Matrix structure.
 * Allocates memory for the matrix data based on the given rows and columns.
 *
 * @param filename The path to the HDF5 file.
 * @param dataset The name of the dataset in the HDF5 file.
 * @param matrix Pointer to the Matrix structure to store the data.
 */
void import_matrix_from_file(const char *filename, Matrix *matrix);

/**
 * Computes the minimum size for brute-force search based on the number of queries.
 *
 * @param num_query The number of query points.
 * @return The minimum size for brute-force search.
 */
int compute_min_size(int num_query);

/**
 * Swaps two elements in an integer array.
 *
 * @param arr The array of integers.
 * @param a The index of the first element to swap.
 * @param b The index of the second element to swap.
 */
void swap(int *arr, int a, int b);

/**
 * Swaps two elements in a float array.
 *
 * @param arr The array of floats.
 * @param a The index of the first element to swap.
 * @param b The index of the second element to swap.
 */
void swap_float(float *arr, int a, int b);

/**
 * Partition helper function for the quickselect algorithm. Rearranges elements
 * in `dist` and `indices` around a pivot so that all elements less than the
 * pivot are on the left and all greater elements are on the right.
 *
 * @param dist Array of distances.
 * @param indices Array of indices associated with `dist`.
 * @param left The starting index of the partition.
 * @param right The ending index of the partition.
 * @param pivot_index The index of the pivot element.
 * @return The new index of the pivot element.
 */
int quickselect_partition(float *dist, int *indices, int left, int right, int pivot_index);

/**
 * Quickselect algorithm to find the k smallest elements in `dist` and reorder
 * `indices` accordingly. This function modifies `dist` and `indices` in-place.
 *
 * @param dist Array of distances.
 * @param indices Array of indices associated with `dist`.
 * @param left The starting index for the selection.
 * @param right The ending index for the selection.
 * @param k The number of smallest elements to select.
 */
void quickselect(float *dist, int *indices, int left, int right, int k);

/**
 * Adjusts the elements of a max-heap to maintain the heap property. Used for
 * organizing distances and associated indices to efficiently find nearest neighbors.
 *
 * @param distances Array of distances.
 * @param indices Array of indices associated with `distances`.
 * @param heap_size The size of the heap.
 * @param root The index of the root element to start heapifying from.
 */
void heapify(float *distances, int *indices, int heap_size, int root);


// Pthread structs 

/**
 * Struct to store arguments for the recursive k-NN search function.
 */
typedef struct
{
    const Matrix *Q;
    int *idx;
    float *dst;
    int k;
    int MIN_SIZE;
    int sampling_reduction;
    int candidate_reduction;
    int CORPUS_CHUNK_SIZE;
    int HEIGHT;
    int depth;
    int mid_offset;
    float *squared_norms;

} recursionArgs;

/**
 * Struct to store arguments for the sampled points k-NN function.
 */
typedef struct
{
    const Matrix *Q1;
    const Matrix *Q2;
    int k;
    int start;
    int end;
    int *idx;
    float *dst;
    const int *sample_idx;
    float *squared_norms;
    int startQ1;
    int startQ2;
    float *correct_dist;
    int *correct_idx;

} sampled_points_Args;

/**
 * Struct to store arguments for the non-sampled point ANN function.
 */
typedef struct
{
    const Matrix *Q1;
    const Matrix *Q2;
    int start;
    int end;
    int k;
    int N;
    int *idx;
    float *dst;
    const int *sample_idx;
    int sample_size;
    float *squared_norms;
    int startQ1;
    int startQ2;
    const bool *is_sampled_map;
    
} non_sampled_Args;



#endif // UTILITIES_H
