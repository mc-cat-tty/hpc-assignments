#include <iostream>
#include "vector"
#include "cmath"
#include "chrono"
#include "correlation.h"
#include "omp.h"
#include <time.h>
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

using namespace std;
using namespace chrono;

template <typename T>
struct Mat
{
    size_t cols_, rows_;
    vector<T> data_;

    Mat(size_t cols, size_t rows) : cols_(cols), rows_(rows)
    {
        data_ = vector<T>(rows * cols, 0);
    }

    T &operator()(size_t r, size_t c)
    {
        return data_[r * cols_ + c];
    }

    const T &operator()(size_t r, size_t c) const
    {
        return data_[r * cols_ + c];
    }

    auto operator&()
    {
        return data_.data();
    }

    size_t size()
    {
        return rows_ * cols_;
    }
    void zeros()
    {
        data_ = vector<T>(rows_ * cols_, 0);
    }
};

template <typename T>
static void init_array(Mat<T> &data)
{
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            data(i, j) = ((DATA_TYPE)i * j) / M;
}

template <typename T>
static void mean_(Mat<T> &data, vector<T> &mean, T float_n)
{
    for (int j = 0; j < M; j++)
    {
        mean[j] = 0.0;
        for (int i = 0; i < N; i++)
            mean[j] += data(i, j);
        mean[j] /= float_n;
    }
}

template <typename T>
static void stddev_(Mat<T> &data, vector<T> &mean, vector<T> &stddev, T float_n)
{
    DATA_TYPE eps = 0.1f;

    for (size_t j = 0; j < M; j++)
    {
        stddev[j] = 0.0;
        for (size_t i = 0; i < N; i++)

            stddev[j] += (data(i, j) - mean[j]) * (data(i, j) - mean[j]);
        stddev[j] /= float_n;
        stddev[j] = sqrt(stddev[j]);
        /* The following in an inelegant but usual way to handle
           near-zero std. dev. values, which below would cause a zero-
           divide. */
        stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    }
}

template <typename T>
static void center_reduce_(Mat<T> &data, vector<T> &mean, vector<T> &stddev, T float_n)
{
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
        {
            data(i, j) -= mean[j];
            data(i, j) /= sqrt(float_n) * stddev[j];
        }
}

template <typename T>
static void compute_corr_(Mat<T> &data, Mat<T> &symmat, T float_n)
{
    size_t j1, j2, i;
    /* Calculate the m * m correlation matrix. */
    for (j1 = 0; j1 < M - 1; j1++)
    {
        symmat(j1, j1) = 1.0;
        for (j2 = j1 + 1; j2 < M; j2++)
        {
            symmat(j1, j2) = 0.0;
            for (i = 0; i < N; i++)
                symmat(j1, j2) += (data(i, j1) * data(i, j2));
            symmat(j2, j1) = symmat(j1, j2);
        }
    }
    symmat(N - 1, N - 1) = 1.0;
}

template <typename T>
static void compute_corr_loop_interchange_not_optimized_(Mat<T> &data, Mat<T> &symmat, T float_n)
{

    for (size_t j1 = 0; j1 < M - 1; j1++)
    {
        symmat(j1, j1) = 1.0;
        for (size_t j2 = j1 + 1; j2 < M; j2++)
            symmat(j1, j2) = 0.0;
    }

    for (size_t i = 0; i < N; i++)
        for (size_t j1 = 0; j1 < M - 1; j1++)
            for (size_t j2 = j1 + 1; j2 < M; j2++)
                symmat(j1, j2) += (data(i, j1) * data(i, j2));

    for (size_t j1 = 0; j1 < M - 1; j1++)
        for (size_t j2 = j1 + 1; j2 < M; j2++)
            symmat(j2, j1) = symmat(j1, j2);
    symmat(M - 1, M - 1) = 1.0;
}

template <typename T>
static void compute_corr_loop_interchange_task_opt_(Mat<T> &data, Mat<T> &symmat, T float_n)
{
    cout << "warning: for some reason task based seems not working\n";
    size_t i, j1, j2;

#pragma omp task
    for (i = 0; i < N; i++)
        symmat(i, i) = 1.0;

    for (j1 = 0; j1 < M - 1; j1++)
#pragma omp task
#pragma omp simd
        for (j2 = j1 + 1; j2 < M; j2++)
            symmat(j1, j2) = 0.0;

    int unroll_size_ = 4;
    int blocks = N / unroll_size_;
#pragma omp taskwait

    for (size_t i = 0; i < blocks; i += 1)
#pragma omp task
        for (j1 = 0; j1 < M - 1; j1++)
#pragma omp simd
            for (j2 = j1 + 1; j2 < M; j2++)
            {
                size_t idx = i * unroll_size_;
                symmat(j1, j2) += (data(idx, j1) * data(idx, j2));
                symmat(j1, j2) += (data(idx + 1, j1) * data(idx + 1, j2));
                symmat(j1, j2) += (data(idx + 2, j1) * data(idx + 2, j2));
                symmat(j1, j2) += (data(idx + 3, j1) * data(idx + 3, j2));
            }
#pragma omp taskwait

    for (size_t i = unroll_size_ * blocks; i < N; i++)
#pragma omp task
        for (size_t j1 = 0; j1 < M - 1; j1++)
#pragma omp simd
            for (size_t j2 = j1 + 1; j2 < M; j2++)
                symmat(j1, j2) += (data(i, j1) * data(i, j2));
#pragma omp taskwait

    for (size_t j1 = 0; j1 < M - 1; j1++)
#pragma omp task
#pragma omp simd
        for (size_t j2 = j1 + 1; j2 < M; j2++)
            symmat(j2, j1) = symmat(j1, j2);

    symmat(M - 1, M - 1) = 1.0;
}

template <typename T>
static void compute_corr_loop_interchange_parallel_opt_(Mat<T> &data, Mat<T> &symmat, T float_n)
{
#pragma omp parallel for
    for (size_t j1 = 0; j1 < M - 1; j1++)
    {
        symmat(j1, j1) = 1.0;
        for (size_t j2 = j1 + 1; j2 < M; j2++)
            symmat(j1, j2) = 0.0;
    }

    int unroll_size_ = 4;
    int blocks = N / unroll_size_;
    for (size_t i = 0; i < blocks; i += 1)
#pragma omp parallel for schedule(dynamic)
        for (size_t j1 = 0; j1 < M - 1; j1++)
#pragma omp simd
            for (size_t j2 = j1 + 1; j2 < M; j2++)
            {
                size_t idx = i * unroll_size_;
                symmat(j1, j2) += (data(idx, j1) * data(idx, j2));
                symmat(j1, j2) += (data(idx + 1, j1) * data(idx + 1, j2));
                symmat(j1, j2) += (data(idx + 2, j1) * data(idx + 2, j2));
                symmat(j1, j2) += (data(idx + 3, j1) * data(idx + 3, j2));
            }

    for (size_t i = unroll_size_ * blocks; i < N; i++)
        for (size_t j1 = 0; j1 < M - 1; j1++)
            for (size_t j2 = j1 + 1; j2 < M; j2++)
                symmat(j1, j2) += (data(i, j1) * data(i, j2));

#pragma omp parallel for
    for (size_t j1 = 0; j1 < M - 1; j1++)
#pragma omp simd
        for (size_t j2 = j1 + 1; j2 < M; j2++)
            symmat(j2, j1) = symmat(j1, j2);

    symmat(M - 1, M - 1) = 1.0;
}

template <typename T>
ostream &operator<<(ostream &os, const Mat<T> &data)
{
    for (int r = 0; r < data.rows_; ++r)
    {
        for (int c = 0; c < data.cols_; ++c)
        {
            cout << data(r, c) << " ";
        }
        cout << endl;
    }
    return os;
}

template <typename T>
static void hash_(Mat<T> &symmat)
{
    double hash_ = 0.;
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < M; j++)
            hash_ += symmat(i, j);
    }
    printf("The computed hash: %f\n", hash_);
}

struct Timer
{
    time_point<steady_clock> start_ = steady_clock::now();
    time_point<steady_clock> stop_ = steady_clock::now();
    string task_name_;

public:
    explicit Timer(string task_name) : task_name_(task_name) {}

    void start()
    {
        start_ = steady_clock::now();
    }
    void start(string task_name)
    {
        task_name_ = task_name;
        start_ = steady_clock::now();
    }
    void stop()
    {
        stop_ = steady_clock::now();
        duration<double> elapsed_ms = stop_ - start_;
        cout << "Elapsed time for " << task_name_ << ": " << (elapsed_ms.count()) << "s" << endl;
    }
};

template <typename T>
static void kernel_correlation(size_t m, size_t n, DATA_TYPE float_n, Mat<T> &data, Mat<T> &symmat,
                               vector<T> &mean, vector<T> &stddev)
{
    mean_(data, mean, float_n);
    stddev_(data, mean, stddev, float_n);
    center_reduce_(data, mean, stddev, float_n);
    compute_corr_(data, symmat, float_n);
}

template <typename T>
static void kernel_correlation_optimized(size_t m, size_t n, DATA_TYPE float_n, Mat<T> &data, Mat<T> &symmat,
                                         vector<T> &mean, vector<T> &stddev)
{
    Timer t("Corr");

    // t.start("Mean");/
    mean_(data, mean, float_n);
    // t.stop();

    // t.start("Std Deviation");
    stddev_(data, mean, stddev, float_n);
    // t.stop();

    // t.start("Center Reduce");
    center_reduce_(data, mean, stddev, float_n);
    // t.stop();

#ifdef LOOP_OPT
    t.start("Loop Opt Corr");
    compute_corr_loop_interchange_not_optimized_(data, symmat, float_n);
#endif
#ifdef TASK_OPT
    t.start("Task Opt Corr");
#pragma omp parallel
    {
#pragma omp master
        compute_corr_loop_interchange_task_opt_(data, symmat, float_n);
    }
#endif
#ifdef PARALLEL_OPT
    t.start("parallel Opt Corr");
    compute_corr_loop_interchange_parallel_opt_(data, symmat, float_n);
#endif
    t.stop();
    // compute_corr_(data, symmat, float_n);
}

template <typename T>
__global__ void corr_kernel_(T *data, T *symmat, size_t height, size_t width)
{
    const size_t row = threadIdx.y + blockDim.y * blockIdx.y;
    const size_t col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < (height) and col < (width))
    {
        for (size_t j2 = col + 1; j2 < width; j2++)
            // symmat[col * width + j2] += (data[row * width + col] * data[row * width + j2]);
            atomicAdd(&(symmat[col * width + j2]), (data[row * width + col] * data[row * width + j2]));
    }
}

template <typename T>
__global__ void corr_kernel_2_(T *data, T *symmat, size_t height, size_t width)
{

    const size_t row = threadIdx.y + blockDim.y * blockIdx.y;
    const size_t col = threadIdx.x + blockDim.x * blockIdx.x;
    const T v = data[row * width + col];
    size_t col_start = threadIdx.x + 1;
    __shared__ T data_shr[BLOCK_SIZE][BLOCK_SIZE];
    if (row < (height) and col < (width))
    {

        for (size_t i = blockIdx.x; i < (width - 1 + BLOCK_SIZE) / BLOCK_SIZE; i++)
        {
            size_t col_offset = threadIdx.x + i * BLOCK_SIZE;

            data_shr[threadIdx.y][threadIdx.x] = (col_offset < width) ? data[row * width + col_offset] : 0;
            __syncthreads();

            size_t upper_bound = (i == (((width - 1 + BLOCK_SIZE) / BLOCK_SIZE) - 1)) ? width - i + BLOCK_SIZE : BLOCK_SIZE;
            for (size_t j = col_start; j < BLOCK_SIZE and ( i * BLOCK_SIZE + j) < width; j++)
            {
                atomicAdd(&(symmat[col * width + (j + i * blockDim.x)]), (v * data_shr[threadIdx.y][j]));
            }
            col_start = 0;
            __syncthreads();
        }
    }
}

int main()
{
    DATA_TYPE float_n = 1.2;
    Mat<DATA_TYPE> data(M, N);
    init_array(data);
    vector<DATA_TYPE> mean(M, 0);
    vector<DATA_TYPE> stddev(M, 0);
    Mat<DATA_TYPE> symmat(M, M);
    Timer t("correlation");
    struct timespec rt[2];

#ifdef BASELINE
    t.start("baseline correlation");
    kernel_correlation(M, N, float_n, data, symmat, mean, stddev);
    t.stop();
    hash_(symmat);
    mean = vector<DATA_TYPE>(M, 0);
    stddev = vector<DATA_TYPE>(M, 0);
    symmat.zeros();
    init_array(data);
#elif defined(LOOP_OPT) or defined(TASK_OPT) or defined(PARALLEL_OPT)
    t.start("Correlation Optimized");
    kernel_correlation_optimized(M, N, float_n, data, symmat, mean, stddev);
    t.stop();
    hash_(symmat);
#endif

#ifdef CUDA
    symmat.zeros();
    clock_gettime(CLOCK_REALTIME, rt + 0);
    DATA_TYPE *data_d, *symmat_d;
    cudaMalloc((void **)&data_d, sizeof(DATA_TYPE) * M * N);
    cudaMalloc((void **)&symmat_d, sizeof(DATA_TYPE) * N * N);

    cudaMemcpy(data_d, &data, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);
    cudaMemset(symmat_d, 0, sizeof(DATA_TYPE) * N * N);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(((N - 1 + BLOCK_SIZE) / BLOCK_SIZE), ((M - 1 + BLOCK_SIZE) / BLOCK_SIZE));
    t.start("GPU CORR");
    corr_kernel_2_<<<gridDim, blockDim>>>(data_d, symmat_d, M, N);
    cudaMemcpy(&symmat, symmat_d, sizeof(DATA_TYPE) * M * N, cudaMemcpyDeviceToHost);
    t.stop();

    clock_gettime(CLOCK_REALTIME, rt + 1);
    double wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
    for (size_t j1 = 0; j1 < M - 1; j1++)
    {
        symmat(j1, j1) = 1.0;
        for (size_t j2 = j1 + 1; j2 < M; j2++)
            symmat(j2, j1) = symmat(j1, j2);
    }
    symmat(M - 1, M - 1) = 1.0;
    hash_(symmat);
    printf("corr_kernel_ (GPU): %9.3f sec %9.1f GFLOPS\n", wt, (float)N * (M - 1) * M / (1.0e9 * wt * 2.0));
#endif

    return 0;
}
