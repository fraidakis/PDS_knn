#include <stdatomic.h>
#include <pthread.h>
#include "knn.h"

#define MAX_THREADS 18 // Maximum number of threads to use
atomic_int thread_count = 0;
pthread_mutex_t same_idx_mutex = PTHREAD_MUTEX_INITIALIZER; // Mutex for updating same sampled indices

/**
 * Recursively divides the dataset and finds the k nearest neighbors for query points.
 * Utilizes parallelism for efficiency and handles smaller subsets directly with
 * `knnsearch_chunked`.
 */
void knnsearch_recursive(const Matrix *Q, int *idx, float *dst, int k, int MIN_SIZE, int sampling_reduction, int candidate_reduction, int CORPUS_CHUNK_SIZE, int HEIGHT, int depth, int mid_offset, float *squared_norms)
{
    if (Q->rows <= MIN_SIZE)
    {
        // Base case: Process this subset directly
        knnsearch_chunked(Q, Q, k, idx, dst, CORPUS_CHUNK_SIZE, squared_norms, mid_offset);
        return;
    }
    else
    {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 1024 * 1024); // 1 MB stack size

        // Create threads for recursive calls
        pthread_t thread1, thread2;

        int mid = Q->rows / 2;
        Matrix Q1 = {Q->data, mid, Q->cols};                           // First half of Q
        Matrix Q2 = {Q->data + mid * Q->cols, Q->rows - mid, Q->cols}; // Second half of Q

        if (thread_count < MAX_THREADS / 2)
        {
            // Prepare thread arguments for both halves
            recursionArgs args1 = {&Q1, idx, dst, k, MIN_SIZE, sampling_reduction, candidate_reduction, CORPUS_CHUNK_SIZE, HEIGHT, depth + 1, mid_offset, squared_norms};
            recursionArgs args2 = {&Q2, idx + mid * k, dst + mid * k, k, MIN_SIZE, sampling_reduction, candidate_reduction, CORPUS_CHUNK_SIZE, HEIGHT, depth + 1, mid_offset + mid, squared_norms};

            pthread_create(&thread1, &attr, knnsearch_thread, &args1);
            pthread_create(&thread2, &attr, knnsearch_thread, &args2);

            // Wait for both threads to complete
            pthread_join(thread1, NULL);
            pthread_join(thread2, NULL);
        }
        else
        {
            knnsearch_recursive(&Q1, idx, dst, k, MIN_SIZE, sampling_reduction, candidate_reduction, CORPUS_CHUNK_SIZE, HEIGHT, depth + 1, mid_offset, squared_norms);
            knnsearch_recursive(&Q2, idx + mid * k, dst + mid * k, k, MIN_SIZE, sampling_reduction, candidate_reduction, CORPUS_CHUNK_SIZE, HEIGHT, depth + 1, mid_offset + mid, squared_norms);
        }

        int sample_size = Q1.rows / sampling_reduction + 1;
        int *sample_idx = (int *)malloc(sample_size * sizeof(int));

        for (int i = 0; i < sample_size; i++)
            sample_idx[i] = rand() % Q1.rows;

        // Array to store correct knn of sampled points
        int *correct_idx = (int *)malloc(2 * sample_size * k * sizeof(int));
        float *correct_dst = (float *)malloc(2 * sample_size * k * sizeof(float));

        // Refine neighbors by analyzing sampled and non-sampled points (temporarily store knn only from other subset)
        sampled_points_knn(&Q1, &Q2, k, idx, dst, sample_idx, sample_size, squared_norms, mid_offset, mid_offset + mid, correct_dst, correct_idx);
        sampled_points_knn(&Q2, &Q1, k, idx + mid * k, dst + mid * k, sample_idx, sample_size, squared_norms, mid_offset + mid, mid_offset, correct_dst + sample_size * k, correct_idx + sample_size * k);

        int N = sample_size / candidate_reduction + 1;

        non_sampled_points_ann(&Q1, &Q2, k, N, idx, dst, sample_idx, sample_size, squared_norms, mid_offset, mid_offset + mid);
        non_sampled_points_ann(&Q2, &Q1, k, N, idx + mid * k, dst + mid * k, sample_idx, sample_size, squared_norms, mid_offset + mid, mid_offset);

        restore_correct_knn(idx, dst, sample_idx, sample_size, correct_dst, correct_idx, k, mid);

        free(sample_idx);
        free(correct_idx);
        free(correct_dst);
    }
}

/**
 * Thread function for performing the k-NN search on a subset of the dataset.
 */
void *knnsearch_thread(void *arg)
{
    atomic_fetch_add(&thread_count, 1);

    recursionArgs *args = (recursionArgs *)arg;
    knnsearch_recursive(args->Q, args->idx, args->dst, args->k, args->MIN_SIZE, args->sampling_reduction, args->candidate_reduction, args->CORPUS_CHUNK_SIZE, args->HEIGHT, args->depth, args->mid_offset, args->squared_norms);

    atomic_fetch_add(&thread_count, -1);

    return NULL;
}

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

                float dist = squared_norms[q + startQ] - 2 * chunk_dot_products[i] + squared_norms[corpus_index + startQ];

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
 * Find the k nearest neighbors for sampled points in a subset.
 * This is done by splitting the sampled points into chunks and processing them in parallel.
 */
void sampled_points_knn(const Matrix *Q1, const Matrix *Q2, int k, int *idx, float *dst, const int *sample_idx, int sample_size, float *squared_norms, int startQ1, int startQ2, float *correct_dst, int *correct_idx)
{
    int num_threads = MAX_THREADS; // Adjust based on the system
    pthread_t threads[num_threads];
    sampled_points_Args thread_args[num_threads];

    int chunk_size = (sample_size + num_threads - 1) / num_threads;

    // Create threads
    for (int t = 0; t < num_threads; t++)
    {
        int start = t * chunk_size;
        int end = (t + 1) * chunk_size > sample_size ? sample_size : (t + 1) * chunk_size;

        thread_args[t] = (sampled_points_Args){Q1, Q2, k, start, end, idx, dst, sample_idx, squared_norms, startQ1, startQ2, correct_dst, correct_idx};
        pthread_create(&threads[t], NULL, process_sampled_points, &thread_args[t]);
    }

    // Wait for threads to finish
    for (int t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }
}

/**
 * Thread function for processing sampled points in parallel.
 */
void *process_sampled_points(void *args)
{
    sampled_points_Args *thread_args = (sampled_points_Args *)args;

    for (int i = thread_args->start; i < thread_args->end; i++)
    {
        float *sample_distances = malloc(thread_args->k * sizeof(float));
        int *sample_indices = (int *)malloc(thread_args->k * sizeof(int));

        int q_idx = thread_args->sample_idx[i];

        // Initialize a max-heap for sampled point distances
        for (int l = 0; l < thread_args->k; l++)
        {
            sample_distances[l] = FLT_MAX;
            sample_indices[l] = -1;

            thread_args->correct_dist[i * thread_args->k + l] = thread_args->dst[q_idx * thread_args->k + l];
            thread_args->correct_idx[i * thread_args->k + l] = thread_args->idx[q_idx * thread_args->k + l];
        }

        // Compute distances for the sampled query point to all points in Q2
        for (int j = 0; j < thread_args->Q2->rows; j++)
        {
            float dist = thread_args->squared_norms[q_idx + thread_args->startQ1] - 2 * cblas_sdot(thread_args->Q1->cols, &thread_args->Q1->data[q_idx * thread_args->Q1->cols], 1, &thread_args->Q2->data[j * thread_args->Q2->cols], 1) + thread_args->squared_norms[thread_args->startQ2 + j];
            // Update heap if the current distance is smaller
            if (dist < sample_distances[0])
            {
                sample_distances[0] = dist;
                sample_indices[0] = j;
                heapify(sample_distances, sample_indices, thread_args->k, 0);
            }
        }

        // Store the k closest neighbors for the sampled query point
        for (int l = 0; l < thread_args->k; l++)
        {
            // pthread_mutex_lock(&same_idx_mutex);
            thread_args->dst[q_idx * thread_args->k + l] = sample_distances[l];
            thread_args->idx[q_idx * thread_args->k + l] = (sample_indices[l] == -1) ? thread_args->startQ2 : sample_indices[l] + thread_args->startQ2; // Non initialized indices
            // pthread_mutex_unlock(&same_idx_mutex);
        }

        // If any of the new knn is closer than the old knn, update the correct knn
        for (int l = 0; l < thread_args->k; l++)
        {
            if (sample_distances[l] < thread_args->correct_dist[i * thread_args->k])
            {
                // Insert the new distance into the thread_args->correct_dist heap
                thread_args->correct_dist[i * thread_args->k] = sample_distances[l];
                thread_args->correct_idx[i * thread_args->k] = thread_args->idx[q_idx * thread_args->k + l];

                // Re-heapify the thread_args->correct_dist and thread_args->correct_idx arrays
                for (int heap_idx = thread_args->k / 2 - 1; heap_idx >= 0; heap_idx--)
                {
                    heapify(thread_args->correct_dist + i * thread_args->k, thread_args->correct_idx + i * thread_args->k, thread_args->k, heap_idx);
                }
            }
        }

        free(sample_distances);
        free(sample_indices);
    }

    return NULL;
}

/**
 * Finds approximate neighbors for non-sampled points using the neighborhoods
 * of their closest sampled neighbors. This is done in parallel for efficiency.
 */
void non_sampled_points_ann(const Matrix *Q1, const Matrix *Q2, int k, int N, int *idx, float *dst, const int *sample_idx, int sample_size, float *squared_norms, int startQ1, int startQ2)
{
    // Mark sampled points in a boolean array for quick lookup
    bool *is_sampled_map = (bool *)calloc(Q1->rows, sizeof(bool));
    for (int i = 0; i < sample_size; i++)
    {
        is_sampled_map[sample_idx[i]] = true;
    }

    int num_threads = MAX_THREADS; // Adjust the number of threads based on your system
    pthread_t threads[num_threads];
    non_sampled_Args thread_args[num_threads];

    int chunk_size = (Q1->rows + num_threads - 1) / num_threads; // Divide work among threads

    // Create threads
    for (int t = 0; t < num_threads; t++)
    {
        int start = t * chunk_size;
        int end = (t + 1) * chunk_size > Q1->rows ? Q1->rows : (t + 1) * chunk_size;
        thread_args[t] = (non_sampled_Args){Q1, Q2, start, end, k, N, idx, dst, sample_idx, sample_size, squared_norms, startQ1, startQ2, is_sampled_map};
        pthread_create(&threads[t], NULL, process_non_sampled_points, &thread_args[t]);
    }

    // Wait for threads to finish
    for (int t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }

    free(is_sampled_map);
}

/**
 * Thread function for processing non-sampled points in parallel.
 */
void *process_non_sampled_points(void *args)
{
    non_sampled_Args *thread_args = (non_sampled_Args *)args;

    for (int y = thread_args->start; y < thread_args->end; y++)
    {
        if (!thread_args->is_sampled_map[y])
        {
            non_sampled_point_ann(
                thread_args->Q1, thread_args->Q2, y, thread_args->k, thread_args->N,
                thread_args->idx, thread_args->dst, thread_args->sample_idx,
                thread_args->sample_size, thread_args->squared_norms,
                thread_args->startQ1, thread_args->startQ2);
        }
    }

    return NULL;
}

/**
 * Approximates the nearest neighbors for a specific non-sampled point.
 * This includes finding the closest sampled points, collecting their neighbors
 * as candidates, and refining the candidate set to determine the top-k neighbors.
 */
void non_sampled_point_ann(const Matrix *Q1, const Matrix *Q2, int non_sampled_idx, int k, int N, int *idx, float *dst, const int *sample_idx, int sample_size, float *squared_norms, int startQ1, int startQ2)
{
    N = (sample_size < N) ? sample_size : N;
    // Limit N to the number of sampled points

    // Step 1: Identify N closest sampled points to the non-sampled point
    int *closest_sampled_points = (int *)malloc(N * sizeof(int));
    find_N_closest_sampled_points(Q1, non_sampled_idx, N, sample_idx, sample_size, closest_sampled_points, squared_norms, startQ1);

    // Step 2: Collect neighbors of the closest sampled points as candidates
    int *candidate_neighbors = (int *)malloc(N * k * sizeof(int));
    get_candidate_neighbors(idx, k, N, closest_sampled_points, candidate_neighbors);

    // Step 3: Refine the candidate set to determine the top-k neighbors
    refine_neighborhood(Q1, Q2, non_sampled_idx, k, N, &idx[non_sampled_idx * k], &dst[non_sampled_idx * k], candidate_neighbors, squared_norms, startQ1, startQ2);

    free(candidate_neighbors);
    free(closest_sampled_points);
}

/**
 * Finds the N closest sampled points to a given non-sampled query point.
 * Efficiently calculates distances using precomputed squared norms and dot products.
 * A max-heap is used to maintain the N closest sampled points.
 */
void find_N_closest_sampled_points(const Matrix *Q1, int non_sampled_idx, int N, const int *sample_idx, int sample_size, int *closest_sampled_points, float *squared_norms, int startQ1)
{
    float *closest_samples_dist = malloc(N * sizeof(float));

    // Initialize the max-heap with FLT_MAX distances
    for (int i = 0; i < N; i++)
    {
        closest_sampled_points[i] = -1;
        closest_samples_dist[i] = FLT_MAX;
    }

    // Iterate over sampled points and update the heap
    for (int i = 0; i < sample_size; i++)
    {
        __builtin_prefetch(&Q1->data[sample_idx[i + 1] * Q1->cols], 0, 3);

        float distance = squared_norms[non_sampled_idx] - 2 * cblas_sdot(Q1->cols, &Q1->data[non_sampled_idx * Q1->cols], 1, &Q1->data[sample_idx[i] * Q1->cols], 1) + squared_norms[sample_idx[i] + startQ1];

        if (distance < closest_samples_dist[0])
        {
            closest_sampled_points[0] = sample_idx[i];
            closest_samples_dist[0] = distance;
            heapify(closest_samples_dist, closest_sampled_points, N, 0);
        }
    }

    free(closest_samples_dist);
}

/**
 * Collects potential neighbor indices from the k nearest neighbors of each
 * closest sampled point, forming a candidate set for further refinement.
 * This ensures candidates from multiple sampled points are included.
 */
void get_candidate_neighbors(const int *idx, int k, int N, int *closest_sampled_points, int *candidate_neighbors)
{
    for (int i = 0; i < N; i++)
    {
        int q_idx = closest_sampled_points[i];
        for (int j = 0; j < k; j++)
        {
            candidate_neighbors[i * k + j] = idx[q_idx * k + j];
        }
    }
}

/**
 * Refines the candidate neighbor set for a non-sampled point by calculating
 * precise distances to each candidate and updating the top-k neighbors using a max-heap.
 * It ensures only the closest candidates are retained and avoids duplicate processing.
 */
void refine_neighborhood(const Matrix *Q1, const Matrix *Q2, int non_sampled_idx, int k, int N, int *non_sampled_neighbors_idx, float *non_sampled_neighbors_dst, int *candidate_neighbors, float *squared_norms, int startQ1, int startQ2)
{
    // Hash map to track already-checked candidates
    bool *is_checked_map = (bool *)calloc(2 * Q2->rows + 1, sizeof(bool));
    if (is_checked_map == NULL)
    {
        fprintf(stderr, "Memory allocation failed for is_checked_map\n");
        exit(EXIT_FAILURE);
    }

    // Build a max-heap for the current neighbors
    for (int i = k / 2 - 1; i >= 0; i--)
    {
        heapify(non_sampled_neighbors_dst, non_sampled_neighbors_idx, k, i);
    }

    // Process each candidate
    for (int i = 0; i < N * k; i++)
    {
        __builtin_prefetch(&Q1->data[non_sampled_idx * Q1->cols], 0, 3); // Prefetch Q1 row

        if (i % 2 == 0)
        {
            __builtin_prefetch(&Q2->data[(candidate_neighbors[i + 1] - startQ2) * Q2->cols], 0, 3); // Prefetch Q2 row
            __builtin_prefetch(&Q2->data[(candidate_neighbors[i + 2] - startQ2) * Q2->cols], 0, 3); // Prefetch Q2 row
        }

        int q2_idx = candidate_neighbors[i];
        int neighbor_idx = q2_idx - startQ2;

        // Skip if already processed
        if (is_checked_map[neighbor_idx])
            continue;

        is_checked_map[neighbor_idx] = true;

        float dist_sq = squared_norms[non_sampled_idx + startQ1] - 2 * cblas_sdot(Q1->cols, &Q1->data[non_sampled_idx * Q1->cols], 1, &Q2->data[(q2_idx - startQ2) * Q2->cols], 1) + squared_norms[q2_idx];

        // Insert into the heap if closer than the farthest neighbor in the top-k list
        if (dist_sq < non_sampled_neighbors_dst[0])
        {
            non_sampled_neighbors_dst[0] = dist_sq;
            non_sampled_neighbors_idx[0] = q2_idx;
            heapify(non_sampled_neighbors_dst, non_sampled_neighbors_idx, k, 0);
        }
    }

    free(is_checked_map);
}

// Function to restore correct_idx and correct_dist to idx and dst of sampled points
void restore_correct_knn(int *idx, float *dst, const int *sample_idx, int sample_size, const float *correct_dst, const int *correct_idx, int k, int mid)
{
    for (int i = 0; i < sample_size; i++)
    {
        int q_idx = sample_idx[i]; // Index of the sampled query point
        for (int j = 0; j < k; j++)
        {
            // Update idx and dst for sampled query points in Q1
            idx[q_idx * k + j] = correct_idx[i * k + j];
            dst[q_idx * k + j] = correct_dst[i * k + j];

            // Update idx and dst for sampled query points in Q2
            idx[(q_idx + mid) * k + j] = correct_idx[(i + sample_size) * k + j];
            dst[(q_idx + mid) * k + j] = correct_dst[(i + sample_size) * k + j];
        }
    }
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
