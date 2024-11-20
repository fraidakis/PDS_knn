#ifndef KNN_H
#define KNN_H

#include <stdbool.h>
#include <math.h>
#include <float.h>
#include "utilities.h"  

/**
 * @brief Recursively performs the k-Nearest Neighbors (k-NN) search on a dataset.
 *
 * @param Q                     Pointer to the query matrix.
 * @param idx                   Pointer to an array where neighbor indices will be stored.
 * @param dst                   Pointer to an array where neighbor distances will be stored.
 * @param k                     Number of nearest neighbors to find.
 * @param MIN_SIZE              Minimum size of the dataset to stop recursion.
 * @param sampling_reduction    Reduction factor for sampling size.
 * @param candidate_reduction   Reduction factor for candidate neighbors.
 * @param CORPUS_CHUNK_SIZE     Size of each corpus chunk for processing.
 * @param HEIGHT                Log2 of the number of recursive divisions.
 * @param depth                 Current depth of recursion.
 * @param mid_offset            Offset used when dividing the dataset.
 * @param squared_norms         Precomputed squared norms of the dataset vectors.
 */
void knnsearch_recursive(const Matrix *Q, int *idx, float *dst, int k, int MIN_SIZE, int sampling_reduction, int candidate_reduction, int CORPUS_CHUNK_SIZE, int HEIGHT, int depth, int mid_offset, float *squared_norms);

/**
 * @brief Performs a chunked k-NN search on smaller subsets of the dataset.
 *
 * @param C                 Pointer to the corpus matrix.
 * @param Q                 Pointer to the query matrix.
 * @param k                 Number of nearest neighbors to find.
 * @param idx               Pointer to an array where neighbor indices will be stored.
 * @param dst               Pointer to an array where neighbor distances will be stored.
 * @param mid_offset        Offset used when dividing the dataset.
 * @param CORPUS_CHUNK_SIZE Size of each corpus chunk for processing.
 * @param squared_norms     Precomputed squared norms of the dataset vectors.
 * @param startQ            Starting index of the query subset.
 */
void knnsearch_chunked(const Matrix *C, const Matrix *Q, int k, int *idx, float *dst, int CORPUS_CHUNK_SIZE, float *squared_norms, int startQ);

/**
 * @brief Approximates neighbors between two subsets of the dataset.
 *
 * @param Q1               Pointer to the first subset matrix.
 * @param Q2               Pointer to the second subset matrix.
 * @param k                Number of nearest neighbors to find.
 * @param idx              Pointer to an array where neighbor indices will be stored.
 * @param dst              Pointer to an array where neighbor distances will be stored.
 * @param sample_idx       Array of sampled indices from Q1.
 * @param sample_size      Number of sampled points.
 * @param squared_norms    Precomputed squared norms of the dataset vectors.
 * @param startQ1          Starting index of the first subset.
 * @param startQ2          Starting index of the second subset.
 * @param correct_idx      Array to store correct indices of sampled points.
 * @param correct_dst      Array to store correct distances of sampled points.
 */
void sampled_points_knn(const Matrix *Q1, const Matrix *Q2, int k, int *idx, float *dst, const int *sample_idx, int sample_size, float *squared_norms, int startQ1, int startQ2, float *correct_dst, int *correct_idx);

/**
 * @brief Approximates neighborhoods for all non-sampled points in the dataset.
 *
 * @param Q1               Pointer to the first subset matrix.
 * @param Q2               Pointer to the second subset matrix.
 * @param k                Number of nearest neighbors to find.
 * @param n                Number of closest sampled points to consider for approximation.
 * @param idx              Pointer to an array where neighbor indices will be stored.
 * @param dst              Pointer to an array where neighbor distances will be stored.
 * @param sample_idx       Array of sampled indices.
 * @param sample_size      Number of sampled points.
 * @param squared_norms    Precomputed squared norms of the dataset vectors.
 * @param startQ1          Starting index of the first subset.
 * @param startQ2          Starting index of the second subset.
 */
void non_sampled_points_ann(const Matrix *Q1, const Matrix *Q2, int k, int n, int *idx, float *dst, const int *sample_idx, int sample_size, float *squared_norms, int startQ1, int startQ2);

/**
 * @brief Wrapper function to approximate neighborhoods for a single non-sampled point.
 *
 * @param Q1               Pointer to the first subset matrix.
 * @param Q2               Pointer to the second subset matrix.
 * @param non_sampled_idx  Index of the non-sampled query point in Q1.
 * @param k                Number of nearest neighbors to find.
 * @param N                Number of closest sampled points to consider for approximation.
 * @param idx              Pointer to an array where neighbor indices will be stored.
 * @param dst              Pointer to an array where neighbor distances will be stored.
 * @param sample_idx       Array of sampled indices.
 * @param sample_size      Number of sampled points.
 * @param squared_norms    Precomputed squared norms of the dataset vectors.
 * @param startQ1          Starting index of the first subset.
 * @param startQ2          Starting index of the second subset.
 */
void non_sampled_point_ann(const Matrix *Q1, const Matrix *Q2, int non_sampled_idx, int k, int N, int *idx, float *dst, const int *sample_idx, int sample_size, float *squared_norms, int startQ1, int startQ2);

/**
 * @brief Finds the n closest sampled points to a non-sampled query point.
 *
 * @param Q1               Pointer to the first subset matrix.
 * @param not_sample_idx   Index of the non-sampled query point in Q1.
 * @param N                Number of closest sampled points to find.
 * @param sample_idx       Array of sampled indices.
 * @param sample_size      Number of sampled points.
 * @param closest_samples  Pointer to an array where the closest sampled indices will be stored.
 * @param squared_norms    Precomputed squared norms of the dataset vectors.
 * @param startQ1          Starting index of the first subset.
 */
void find_N_closest_sampled_points(const Matrix *Q1, int not_sample_idx, int N, const int *sample_idx, int sample_size, int *closest_samples, float *squared_norms, int startQ1);

/**
 * @brief Retrieves candidate neighbors for a non-sampled query point based on closest sampled points.
 *
 * @param idx                   Pointer to the array containing neighbor indices.
 * @param k                     Number of nearest neighbors per sampled point.
 * @param N                     Number of closest sampled points.
 * @param closest_samples       Array of indices of the closest sampled points.
 * @param candidate_neighbors   Output array to store candidate neighbor indices.
 */
void get_candidate_neighbors(const int *idx, int k, int N, int *closest_sampled_points, int *candidate_neighbors);
/**
 * @brief Refines the neighborhood of a non-sampled query point by evaluating candidate neighbors.
 * 
 * @param Q1                         Pointer to the first subset matrix.
 * @param Q2                         Pointer to the second subset matrix.
 * @param non_sampled_idx            Index of the non-sampled query point in Q1.
 * @param k                          Number of nearest neighbors to find.
 * @param N                          Number of candidate sets to evaluate.
 * @param non_sampled_neighbors_idx  Pointer to an array where the final neighbor indices will be stored.
 * @param non_sampled_neighbors_dst  Pointer to an array where the final neighbor distances will be stored.
 * @param candidate_neighbors        Array of candidate neighbor indices.
 * @param squared_norms              Precomputed squared norms of the dataset vectors.
 * @param startQ1                    Starting index of the first subset.
 * @param startQ2                    Starting index of the second subset.
 */
void refine_neighborhood(const Matrix *Q1, const Matrix *Q2, int non_sampled_idx, int k, int N, int *non_sampled_neighbors_idx, float *non_sampled_neighbors_dst, int *candidate_neighbors, float *squared_norms, int startQ1, int startQ2);

/**
 * @brief Restores the correct indices and distances of sampled points to the output arrays.
 * 
 * @param idx           Pointer to the array containing neighbor indices.
 * @param dst           Pointer to the array containing neighbor distances.
 * @param sample_idx    Array of sampled indices.
 * @param sample_size   Number of sampled points.
 * @param correct_dst   Array of correct distances for sampled points.
 * @param correct_idx   Array of correct indices for sampled points.
 * @param k             Number of nearest neighbors to find.
 * @param mid           Offset used when dividing the dataset.
 */
void restore_correct_knn(int *idx, float *dst, const int *sample_idx, int sample_size, const float *correct_dst, const int *correct_idx, int k, int mid);

/**
 * Computes the Euclidean norms (squared magnitudes) for each row vector in a matrix.
 * Stores the result in the `norms` array.
 *
 * @param matrix Pointer to the Matrix structure containing the vectors.
 * @param norms Array where computed norms for each row will be stored.
 */
void compute_norms(const Matrix *matrix, float *norms);



// Pthread functions

/**
 * @brief Thread function for performing the k-NN search on a subset of the dataset.
 *
 * @param arg Pointer to the arguments struct for the recursive k-NN search function.
 * @return void* NULL pointer.
 */
void *knnsearch_thread(void *arg);

/**
 * @brief Thread function for processing sampled points in parallel.
 *
 * @param args Pointer to the arguments struct for the sampled points k-NN function.
 * @return void* NULL pointer.
 */
void *process_sampled_points(void *args);

/**
 * @brief Thread function for processing non-sampled points in parallel.
 *
 * @param args Pointer to the arguments struct for the non-sampled point ANN function.
 * @return void* NULL pointer.
 */
void *process_non_sampled_points(void *args);

#endif // KNN_H
