#include "utilities.h"

/**
 * Returns the current time in seconds.
 * Used for performance timing.
 */
double get_time_in_seconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

void update_file_name(char *filename, const char *path, char *dataset)
{
    if (strcmp(dataset, "mnist") == 0)
    {
        strcpy(filename, path);
        strcat(filename, "mnist-784-euclidean.hdf5");
    }
    else if (strcmp(dataset, "fashion-mnist") == 0)
    {
        strcpy(filename, path);
        strcat(filename, "fashion-mnist-784-euclidean.hdf5");
    }
    else if (strcmp(dataset, "sift") == 0)
    {
        strcpy(filename, path);
        strcat(filename, "sift-128-euclidean.hdf5");
    }
    else
    {
        fprintf(stderr, "Invalid dataset name. Please choose from: mnist, fashion-mnist, sift\n");
        exit(1);
    }
}

/**
 * Updates the parameters based on command line arguments.
 */
void update_parameters(char *argv[], int num_query, int *sampling_reduction, int *candidate_reduction, int *MIN_SIZE)
{
    *sampling_reduction = atoi(argv[2]);
    *candidate_reduction = atoi(argv[3]);
    *MIN_SIZE = compute_min_size(num_query) / atoi(argv[4]);
}

/**
 * Loads both "test" and "train" datasets from an HDF5 file into a Matrix structure.
 */
void import_matrix_from_file(const char *filename, Matrix *matrix)
{
    Matrix test, train;

    // Open the HDF5 file
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
    {
        fprintf(stderr, "Error opening HDF5 file: %s\n", filename);
        exit(1);
    }

    // Open "test" dataset
    hid_t dataset_id = H5Dopen(file_id, "test", H5P_DEFAULT); // Adjust "dataset" to your dataset name
    if (dataset_id < 0)
    {
        fprintf(stderr, "Error opening dataset in HDF5 file: %s\n", filename);
        H5Fclose(file_id);
        exit(1);
    }

    // Get dataset dimensions
    hid_t dataspace_id = H5Dget_space(dataset_id);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

    // Allocate memory for test matrix
    test.data = aligned_alloc(32, dims[0] * dims[1] * sizeof(float));
    test.rows = dims[0];
    test.cols = dims[1];

    // Read the dataset into the test->data
    if (H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, test.data) < 0)
    {
        fprintf(stderr, "Error reading dataset from HDF5 file: %s\n", filename);
        free(test.data);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        exit(1);
    }

    // Close HDF5 handles
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    // Open the HDF5 file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
    {
        fprintf(stderr, "Error opening HDF5 file: %s\n", filename);
        exit(1);
    }

    // Open "train" dataset
    dataset_id = H5Dopen(file_id, "train", H5P_DEFAULT); // Adjust "dataset" to your dataset name
    if (dataset_id < 0)
    {
        fprintf(stderr, "Error opening dataset in HDF5 file: %s\n", filename);
        H5Fclose(file_id);
        exit(1);
    }

    // Get dataset dimensions
    dataspace_id = H5Dget_space(dataset_id);
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

    // Allocate memory for train matrix
    train.data = aligned_alloc(32, dims[0] * dims[1] * sizeof(float));
    train.rows = dims[0];
    train.cols = dims[1];

    // Read the dataset into the train->data
    if (H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, train.data) < 0)
    {
        fprintf(stderr, "Error reading dataset from HDF5 file: %s\n", filename);
        free(train.data);
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        exit(1);
    }

    // Close HDF5 handles
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    // Copy test and train to matrix
    matrix->data = aligned_alloc(32, (test.rows + train.rows) * test.cols * sizeof(float));
    matrix->rows = test.rows + train.rows;
    matrix->cols = test.cols;

    for (int i = 0; i < test.rows; i++)
    {
        for (int j = 0; j < test.cols; j++)
        {
            matrix->data[i * test.cols + j] = test.data[i * test.cols + j];
        }
    }

    for (int i = 0; i < train.rows; i++)
    {
        for (int j = 0; j < train.cols; j++)
        {
            matrix->data[(i + test.rows) * train.cols + j] = train.data[i * train.cols + j];
        }
    }

    free(test.data);
    free(train.data);
}

/**
 * Computes the minimum size for brute-force search based on the number of queries.
 */
int compute_min_size(int num_query)
{
    while (num_query > 250)
    {
        num_query = num_query / 2;
    }

    return num_query;
}

/**
 * Swaps two elements in an integer array.
 */
void swap(int *arr, int a, int b)
{
    int temp = arr[a];
    arr[a] = arr[b];
    arr[b] = temp;
}

/**
 * Swaps two elements in a float array.
 */
void swap_float(float *arr, int a, int b)
{
    float temp = arr[a];
    arr[a] = arr[b];
    arr[b] = temp;
}

/**
 * Partitions the array around a pivot for quickselect.
 * Elements less than pivot are moved to the left, and greater ones to the right.
 */
int quickselect_partition(float *dist, int *indices, int left, int right, int pivot_index)
{
    float pivot_value = dist[pivot_index];

    // Swap pivot to end
    swap_float(dist, pivot_index, right);
    swap(indices, pivot_index, right);

    int store_index = left;
    for (int i = left; i < right; i++)
    {
        if (dist[i] < pivot_value)
        {
            swap_float(dist, store_index, i);
            swap(indices, store_index, i);
            store_index++;
        }
    }

    // Place pivot in its final position
    swap_float(dist, right, store_index);
    swap(indices, right, store_index);

    return store_index;
}

/**
 * Quickselect algorithm for finding k smallest distances.
 */
void quickselect(float *dist, int *indices, int left, int right, int k)
{
    if (left < right)
    {
        int pivot_index = left + (right - left) / 2;
        pivot_index = quickselect_partition(dist, indices, left, right, pivot_index);

        if (k == pivot_index)
        {
            return;
        }
        else if (k < pivot_index)
        {
            quickselect(dist, indices, left, pivot_index - 1, k);
        }
        else
        {
            quickselect(dist, indices, pivot_index + 1, right, k);
        }
    }
}

/**
 * Heapifies the subtree rooted at `root` in a max-heap.
 */
void heapify(float *distances, int *indices, int heap_size, int root)
{
    int largest = root;
    int left = 2 * root + 1;
    int right = 2 * root + 2;

    if (left < heap_size && distances[left] > distances[largest])
        largest = left;

    if (right < heap_size && distances[right] > distances[largest])
        largest = right;

    if (largest != root)
    {
        // Swap distances and indices
        swap_float(distances, root, largest);
        swap(indices, root, largest);

        heapify(distances, indices, heap_size, largest);
    }
}
