#include <iostream>
#include "vector"
#include "cmath"
#include "chrono"
#include "correlation.h"
#include "omp.h"
#include <time.h>
#include <cuda_runtime.h>
#include <assert.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 32
#endif

#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 32
#endif

using namespace std;
using namespace chrono;

static void stats()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "Shared Memory per Multiprocessor: " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
    }
    int device;
    cudaGetDevice(&device); // Ottieni il dispositivo corrente

    size_t const_mem_size;
    cudaDeviceGetAttribute((int *)&const_mem_size, cudaDevAttrTotalConstantMemory, device);

    std::cout << "Memoria const disponibile: " << const_mem_size << " byte" << std::endl;
    std::cout << "=============================" << std::endl;
}

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

    void t()
    {
        vector<T> new_data(rows_ * cols_);
        // #pragma omp parallel for collapse(2)
        for (int r = 0; r < rows_; r++)
            for (int c = 0; c < cols_; c++)
                new_data[c * rows_ + r] = data_[r * cols_ + c];

        data_ = new_data;
        size_t tmp = rows_;
        rows_ = cols_;
        cols_ = tmp;
    }

    Mat<T> transpose()
    {
        Mat<T> transpose(cols_, rows_);
        for (int c = 0; c < cols_; ++c)
            for (int r = 0; r < rows_; ++r)
                transpose(c, r) = (*this)(r, c);
        return transpose;
    }

    size_t size()
    {
        return rows_ * cols_;
    }

    void zeros()
    {
        data_ = vector<T>(rows_ * cols_, 0);
    }

    void print()
    {
        cout << "[" << rows_ << "," << cols_ << "]\n";
        for (int r = 0; r < rows_; ++r)
        {
            for (int c = 0; c < cols_; ++c)
            {
                cout << data_[r * cols_ + c] << ", ";
            }
            cout << endl;
        }
        cout << endl;
    }
};

template <typename T>
static void init_array(Mat<T> &data)
{
    for (size_t i = 0; i < data.rows_; i++)
        for (size_t j = 0; j < data.cols_; j++)
            data(i, j) = ((T)i * j) / M;
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
    T eps = 0.1f;

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
        cout << "Elapsed time for " << task_name_ << ": " << (elapsed_ms.count()) << " sec" << endl;
    }

    void stop_flops(float flops)
    {
        stop_ = steady_clock::now();
        duration<double> elapsed_ms = stop_ - start_;
        cout << "Elapsed time for " << task_name_ << ": " << (elapsed_ms.count()) << " sec" << "    " << flops / elapsed_ms.count() << " GFLOPS" << endl;
    }
};

template <typename T>
static void kernel_correlation(size_t m, size_t n, T float_n, Mat<T> &data, Mat<T> &symmat,
                               vector<T> &mean, vector<T> &stddev)
{
    mean_(data, mean, float_n);
    stddev_(data, mean, stddev, float_n);
    center_reduce_(data, mean, stddev, float_n);
    compute_corr_(data, symmat, float_n);
}

template <typename T>
static void kernel_correlation_optimized(size_t m, size_t n, T float_n, Mat<T> &data, Mat<T> &symmat,
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
__global__ void gemm_v2(T *__restrict__ a, T *__restrict__ b, T *__restrict__ c, size_t n)
{
    __shared__ T as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T bs[BLOCK_SIZE][BLOCK_SIZE];

    const size_t a_row = threadIdx.y + blockIdx.y * blockDim.y;
    //  a_col = threadIdx.x;

    //  b_row = threadIdx.y;
    const size_t b_col = threadIdx.x + blockDim.x * blockIdx.x;

    T accum = (T)0;

    for (int kb = 0; kb < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++kb)
    {
        size_t a_col = threadIdx.x + kb * blockDim.x;
        size_t b_row = threadIdx.y + kb * blockDim.y;

        as[threadIdx.y][threadIdx.x] = (a_row < n && a_col < n) ? a[a_row * n + a_col] : (T)0;
        bs[threadIdx.y][threadIdx.x] = (b_row < n && b_col < n) ? b[b_row * n + b_col] : (T)0;

        __syncthreads();

        for (size_t i = 0; i < BLOCK_SIZE && (kb * BLOCK_SIZE + i) < n; i++)
        {
            accum += as[threadIdx.y][i] * bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (a_row < n && b_col < n)
        c[a_row * n + b_col] = accum;
}

template <typename T>
__global__ void gemm_v3(T *__restrict__ a, T *__restrict__ c, size_t n)
{
    __shared__ T as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T bs[BLOCK_SIZE][BLOCK_SIZE];

    const size_t a_row = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t b_col = threadIdx.x + blockDim.x * blockIdx.x;

    T accum = (T)0;
    if ((b_col + BLOCK_SIZE) >= a_row)
    {
        for (int kb = 0; kb < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++kb)
        {
            size_t a_col = threadIdx.x + kb * blockDim.x;
            size_t b_row = threadIdx.y + kb * blockDim.y;

            as[threadIdx.y][threadIdx.x] = (a_row < n && a_col < n) ? a[a_row * n + a_col] : (T)0;
            bs[threadIdx.y][threadIdx.x] = (b_row < n && b_col < n) ? a[b_col * n + b_row] : (T)0;

            __syncthreads();

            for (size_t i = 0; i < BLOCK_SIZE && (kb * BLOCK_SIZE + i) < n; i++)
                accum += as[threadIdx.y][i] * bs[i][threadIdx.x];

            __syncthreads();
        }

        if (a_row < n && b_col < n)
            c[a_row * n + b_col] = accum;
    }
}

template <typename T>
static void cuda_corr_(size_t m, size_t n, T float_n, Mat<T> &data, Mat<T> &symmat,
                       vector<T> &mean, vector<T> &stddev)
{
    Timer t("general");

    t.start("mean + stddev + center_reduce");
    mean_(data, mean, float_n);
    stddev_(data, mean, stddev, float_n);
    center_reduce_(data, mean, stddev, float_n);
    t.stop();

    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim(((N - 1 + BLOCK_SIZE_X) / BLOCK_SIZE_X), ((M - 1 + BLOCK_SIZE_Y) / BLOCK_SIZE_Y));

    t.start("MemAlloc, data transpose and Cpy[HtoD]");
    T *data_d, *dataT_d, *symmat_d;
    cudaMalloc((void **)&dataT_d, sizeof(T) * M * N);
    cudaMalloc((void **)&symmat_d, sizeof(T) * N * N);

    cudaMemset(symmat_d, 0, sizeof(T) * N * N);
    data.t();
    cudaMemcpy(dataT_d, &data, sizeof(T) * M * N, cudaMemcpyHostToDevice);
    t.stop();

    t.start("gemm_v3(GPU)");
    gemm_v3<<<gridDim, blockDim>>>(dataT_d, symmat_d, N);
    cudaMemcpy(&symmat, symmat_d, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
    t.stop_flops((float)N * (M - 1) * M / (1.0e9 * 2.0));

    for (size_t j1 = 0; j1 < M - 1; j1++)
    {
        symmat(j1, j1) = 1.0;
        for (size_t j2 = j1 + 1; j2 < M; j2++)
            symmat(j2, j1) = symmat(j1, j2);
    }
    symmat(M - 1, M - 1) = 1.0;
}

int main()
{
    DATA_TYPE float_n = 1.2;
    Mat<DATA_TYPE> data(M, N);
    vector<DATA_TYPE> mean(M, 0);
    vector<DATA_TYPE> stddev(M, 0);
    Mat<DATA_TYPE> symmat(M, M);
    Mat<DATA_TYPE> symmat_default(M, M);
    Timer t("general");

#ifdef BASELINE
    mean = vector<DATA_TYPE>(M, 0);
    stddev = vector<DATA_TYPE>(M, 0);
    init_array(data);
    symmat_default.zeros();
    t.start("total baseline corr");
    kernel_correlation(M, N, float_n, data, symmat_default, mean, stddev);
    t.stop();
    hash_(symmat_default);
#endif

#if defined(LOOP_OPT) or defined(TASK_OPT) or defined(PARALLEL_OPT)
    mean = vector<DATA_TYPE>(M, 0);
    stddev = vector<DATA_TYPE>(M, 0);
    init_array(data);
    symmat_default.zeros();
    t.start("Total Corr Optimized ");
    kernel_correlation_optimized(M, N, float_n, data, symmat_default, mean, stddev);
    t.stop();
    hash_(symmat_default);
#endif

#ifdef CUDA

    mean = vector<DATA_TYPE>(M, 0);
    stddev = vector<DATA_TYPE>(M, 0);
    init_array(data);
    symmat.zeros();

    t.start("Total GPU Corr");
    cuda_corr_(M, N, float_n, data, symmat, mean, stddev);
    t.stop();
    hash_(symmat);

#endif
    return 0;
}

// }