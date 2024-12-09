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

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

template <typename T>
class Mat
{
private:
    T *allocator(bool &error, size_t size)
    {
        T *mem;

        if (mem_type_ == UVM_MEM)
        {
            auto res = cudaMallocManaged(&mem, size);
            error = (res != NULL);
        }
        else if (mem_type_ == PINNED_MEM)
        {
            auto res = cudaMallocHost(&mem, size);
            error = (res != NULL);
        }
        else if (mem_type_ == STANDARD_MEM)
        {
            mem = static_cast<T *>(malloc(size));
            error = (mem == nullptr);
        }

        if (not error)
        {
            if (mem_type_ == PINNED_MEM)
                cudaMemset(mem, 0, rows_ * cols_ * sizeof(T));
            else
                memset(mem, 0, rows_ * cols_ * sizeof(T));
        }

        return mem;
    }

    void deallocator(T *mem)
    {
        if (mem_type_ == UVM_MEM)
            cudaFree(data_);
        else if (mem_type_ == PINNED_MEM)
            cudaFreeHost(data_);
        else if (mem_type_ == STANDARD_MEM)
            free(data_);
    }

public:
    size_t cols_, rows_;
    T *data_;
    uint8_t mem_type_;

    Mat(size_t cols, size_t rows, uint8_t mem_type) : cols_(cols), rows_(rows), mem_type_(mem_type)
    {
        bool error;
        data_ = allocator(error, rows_ * cols_ * sizeof(T));

        if (error)
        {
            cout << "Bad allocation for Mat" << endl;
            deallocator(data_);
            exit(1);
        }
    }

    ~Mat()
    {
        deallocator(data_);
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
        return data_;
    }

    void t()
    {
        bool error;
        T *new_data = allocator(error, rows_ * cols_ * sizeof(T));
        if (error)
            cout << "New data memory for transposition not allocated" << endl;

        // #pragma omp parallel for collapse(2)
        for (int r = 0; r < rows_; r++)
            for (int c = 0; c < cols_; c++)
                new_data[c * rows_ + r] = data_[r * cols_ + c];

        deallocator(data_);
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
        memset(data_, 0, rows_ * cols_ * sizeof(T));
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
__global__ void mean_kernel_(T *__restrict__ data, T *__restrict__ mean, T float_n)
{
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < M)
    {
        for (size_t i = 0; i < M; i++)
            mean[idx] += data[idx * M + i];
        mean[idx] /= float_n;
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
__global__ void kernel_stddev_(T *__restrict__ data, T *__restrict__ mean, T *__restrict__ stddev, T float_n)
{
    T eps = 0.1f;
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < M)

    {
        stddev[idx] = 0.0;
        for (size_t i = 0; i < N; i++)
            stddev[idx] += (data[M * idx + i] - mean[idx]) * (data[M * idx + i] - mean[idx]);

        stddev[idx] /= float_n;
        stddev[idx] = sqrt(stddev[idx]);
        stddev[idx] = stddev[idx] <= eps ? 1.0 : stddev[idx];
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
__global__ void kernel_center_reduce_(T *__restrict__ data, T *__restrict__ mean, T *__restrict__ stddev, T float_n)
{

    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < M)

        for (size_t j = 0; j < M; j++)
        {
            data[idx * M + j] -= mean[idx];
            data[idx * M + j] /= sqrt(float_n) * stddev[idx];
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
static void cuda_corr_(size_t m, size_t n, T float_n, Mat<T> &data, Mat<T> &symmat, vector<T> &mean, vector<T> &stddev)
{
    Timer t("general");

    // t.start("mean + stddev + center_reduce");
    // mean_(data, mean, float_n);
    // stddev_(data, mean, stddev, float_n);
    // center_reduce_(data, mean, stddev, float_n);
    // t.stop();

    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim(((N - 1 + BLOCK_SIZE_X) / BLOCK_SIZE_X), ((M - 1 + BLOCK_SIZE_Y) / BLOCK_SIZE_Y));

    t.start("MemAlloc, data transpose and Cpy[HtoD]");

#if MEM_MODEL != UVM_MEM
    T *dataT_d, *symmat_d, *mean_d, *stddev_d;
    cudaMalloc((void **)&dataT_d, sizeof(T) * M * N);
    cudaMalloc((void **)&symmat_d, sizeof(T) * N * N);
    cudaMalloc((void **)&mean_d, sizeof(T) * N);
    cudaMalloc((void **)&stddev_d, sizeof(T) * N);

    cudaMemset(symmat_d, 0, sizeof(T) * N * N);
    cudaMemset(mean_d, 0, sizeof(T) * N);
    cudaMemset(stddev_d, 0, sizeof(T) * N);
#endif

    data.t();

#if MEM_MODEL != UVM_MEM
    cudaMemcpy(dataT_d, &data, sizeof(T) * M * N, cudaMemcpyHostToDevice);
#endif
#if MEM_MODEL != UVM_MEM

    mean_kernel_<<<((N - 1 + BLOCK_SIZE) / BLOCK_SIZE), BLOCK_SIZE>>>(dataT_d, mean_d, float_n);
    kernel_stddev_<<<((N - 1 + BLOCK_SIZE) / BLOCK_SIZE), BLOCK_SIZE>>>(dataT_d, mean_d, stddev_d, float_n);
    kernel_center_reduce_<<<((N - 1 + BLOCK_SIZE) / BLOCK_SIZE), BLOCK_SIZE>>>(dataT_d, mean_d, stddev_d, float_n);
#elif MEM_MODEL == UVM_MEM
    mean_kernel_<<<((N - 1 + BLOCK_SIZE) / BLOCK_SIZE), BLOCK_SIZE>>>(&data, &(mean[0]), float_n);
    kernel_stddev_<<<((N - 1 + BLOCK_SIZE) / BLOCK_SIZE), BLOCK_SIZE>>>(&data, &(mean[0]), &(stddev[0]), float_n);
    kernel_center_reduce_<<<((N - 1 + BLOCK_SIZE) / BLOCK_SIZE), BLOCK_SIZE>>>(&data, &(mean[0]), &(stddev[0]), float_n);
#endif

    t.stop();

    t.start("gemm_v3(GPU)");

#if MEM_MODEL != UVM_MEM
    gemm_v3<<<gridDim, blockDim>>>(dataT_d, symmat_d, N);
#elif MEM_MODEL == UVM_MEM
    gemm_v3<<<gridDim, blockDim>>>(&data, &symmat, N);
#endif

    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();

#if MEM_MODEL != UVM_MEM
    cudaMemcpy(&symmat, symmat_d, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
#endif

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

    Timer t("general");

#ifdef BASELINE
    {
        Mat<DATA_TYPE> data(M, N, STANDARD_MEM);
        vector<DATA_TYPE> mean(M, 0);
        vector<DATA_TYPE> stddev(M, 0);
        Mat<DATA_TYPE> symmat(M, M, STANDARD_MEM);
        init_array(data);
        symmat.zeros();
        t.start("total baseline corr");
        kernel_correlation(M, N, float_n, data, symmat, mean, stddev);
        t.stop();
        hash_(symmat);
    }
#endif

#if defined(LOOP_OPT) or defined(TASK_OPT) or defined(PARALLEL_OPT)
    {
        Mat<DATA_TYPE> data(M, N, STANDARD_MEM);
        vector<DATA_TYPE> mean(M, 0);
        vector<DATA_TYPE> stddev(M, 0);
        Mat<DATA_TYPE> symmat(M, M, STANDARD_MEM);
        init_array(data);
        symmat.zeros();
        t.start("Total Corr Optimized ");
        kernel_correlation_optimized(M, N, float_n, data, symmat, mean, stddev);
        t.stop();
        hash_(symmat);
    }
#endif

#ifdef CUDA
    {
#if MEM_MODEL == UVM_MEM
        cout << "Using UVM memory model" << endl;
#elif MEM_MODEL == PINNED_MEM
        cout << "Using pinned memory model" << endl;
#elif MEM_MODEL == STANDARD_MEM
        cout << "Using standard memory model" << endl;
#endif

        Mat<DATA_TYPE> data(M, N, MEM_MODEL);
        vector<DATA_TYPE> mean(M, 0);
        vector<DATA_TYPE> stddev(M, 0);
        Mat<DATA_TYPE> symmat(M, M, MEM_MODEL);
        init_array(data);
        symmat.zeros();

        t.start("Total GPU Corr");
        cuda_corr_(M, N, float_n, data, symmat, mean, stddev);
        t.stop();
        hash_(symmat);
    }

#endif
    return 0;
}
