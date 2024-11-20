#include "knn.h"

/**
 * Processes the dataset in manageable chunks for efficient k-NN computation.
 * This is used as the base case for the recursive function.
 */
void knnsearch_chunked(const Matrix *C, const Matrix *Q, int k, int *idx, float *dst, int CORPUS_CHUNK_SIZE, float *squared_norms, int startQ)
{
    float *chunk_dot_products = malloc(CORPUS_CHUNK_SIZE * sizeof(float));

    for (int q = 0; q < Q->rows; q++)
    {
        // Initialize the heap (k closest neighbors) for the current query point
        for (int l = 0; l < k; l++)
        {
            dst[q * k + l] = INFINITY;
            idx[q * k + l] = -1;
        }

        // Process the corpus in chunks
        for (int chunk_start = 0; chunk_start < C->rows; chunk_start += CORPUS_CHUNK_SIZE)
        {
            int chunk_size = (chunk_start + CORPUS_CHUNK_SIZE > C->rows) ? C->rows - chunk_start : CORPUS_CHUNK_SIZE;

            // Prefetch next query point for improved memory access
            __builtin_prefetch(&Q->data[(q + 1) * Q->cols], 0, 3);

            // Compute dot products using cblas_sgemm for current chunk
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, chunk_size, 1, Q->cols, 1.0,
                        &C->data[chunk_start * C->cols], C->cols,
                        &Q->data[q * Q->cols], Q->cols, 0.0,
                        chunk_dot_products, 1);

            for (int i = 0; i < chunk_size; i++)
            {
                int corpus_index = chunk_start + i;

                float dist = squared_norms[corpus_index] - 2 * chunk_dot_products[i] + squared_norms[q];

                // If the distance is smaller than the largest in the heap, replace it
                if (dist < dst[q * k] && q != corpus_index)
                {
                    dst[q * k] = dist;
                    idx[q * k] = corpus_index + startQ;

                    // Maintain the heap property for the k nearest neighbors
                    heapify(dst + q * k, idx + q * k, k, 0);
                }
            }
        }
    }

    free(chunk_dot_products);
}

/**
 * Computes the norms (squared magnitudes) for each row vector in the matrix.
 */
void compute_norms(const Matrix *matrix, float *norms)
{
    for (int i = 0; i < matrix->rows; i++)
    {
        norms[i] = cblas_sdot(matrix->cols, &matrix->data[i * matrix->cols], 1, &matrix->data[i * matrix->cols], 1);
    }
}

// Stub function
void knnsearch_recursive(const Matrix *Q __attribute__((unused)), int *idx __attribute__((unused)),
                         float *dst __attribute__((unused)), int k __attribute__((unused)),
                         int MIN_SIZE __attribute__((unused)), int sampling_reduction __attribute__((unused)),
                         int candidate_reduction __attribute__((unused)), int CORPUS_CHUNK_SIZE __attribute__((unused)),
                         int HEIGHT __attribute__((unused)), int depth __attribute__((unused)),
                         int mid_offset __attribute__((unused)), float *squared_norms __attribute__((unused)))
{
    fprintf(stderr, "knnsearch_recursive is not implemented in Sequential.\n");
    exit(EXIT_FAILURE);
}
