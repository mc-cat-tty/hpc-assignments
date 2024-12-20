#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1000. */
#include "correlation.h"


/* Array initialization. */
static void init_array(int m,
                       int n,
                       DATA_TYPE *float_n,
                       DATA_TYPE POLYBENCH_2D(data, M, N, m, n))
{
  int i, j;

  *float_n = 1.2;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      data[i][j] = ((DATA_TYPE)i * j) / M;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m,
                        DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

static void hash_(DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m))
{
  double hash_ = 0.;
  for (size_t i = 0; i < M; i++)
  {
    for (size_t j = 0; j < M; j++)
      hash_ += symmat[i][j];
  }
  printf("The computed hash: %f\n", hash_);
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_correlation(int m, int n,
                               DATA_TYPE float_n,
                               DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
                               DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m),
                               DATA_TYPE POLYBENCH_1D(mean, M, m),
                               DATA_TYPE POLYBENCH_1D(stddev, M, m))
{
  int i, j, j1, j2;

  DATA_TYPE eps = 0.1f;

#define sqrt_of_array_cell(x, j) sqrt(x[j])

  /* Determine mean of column vectors of input data matrix */
  for (j = 0; j < _PB_M; j++)
  {
    mean[j] = 0.0;
    for (i = 0; i < _PB_N; i++)
      mean[j] += data[i][j];
    mean[j] /= float_n;
  }
  /* Determine standard deviations of column vectors of data matrix. */
  for (j = 0; j < _PB_M; j++)
  {
    stddev[j] = 0.0;
    for (i = 0; i < _PB_N; i++)
      stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
    stddev[j] /= float_n;
    stddev[j] = sqrt_of_array_cell(stddev, j);
    /* The following in an inelegant but usual way to handle
       near-zero std. dev. values, which below would cause a zero-
       divide. */
    stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
  }

  /* Center and reduce the column vectors. */
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
    {
      data[i][j] -= mean[j];
      data[i][j] /= sqrt(float_n) * stddev[j];
    }

  /* Calculate the m * m correlation matrix. */
  for (int i = 0; i < _PB_M; i++) {
    symmat[i][i] = 1.0;
  }

  for (int i = 0; i < _PB_N; i++) {
    for (int r = 0; r < _PB_N; r++) {
      for (int c = r+1; c < _PB_M; c++) {
        symmat[r][c] += data[i][r] * data[i][c];
      }
    }
  }

  for (int i = 0; i < _PB_M; i++) {
    for (int j = 0; j < _PB_N; j++) {
      symmat[j][i] = symmat[i][j];
    }
  }
}

static void mean_(int m, int n,
                  DATA_TYPE float_n,
                  DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
                  DATA_TYPE POLYBENCH_1D(mean, M, m))
{
    //#pragma omp target teams num_teams((M*N) / NTHREADS_GPU) thread_limit(NTHREADS_GPU) map(tofrom: mean[0:M]) map(to: data[0:N][0:M], float_n)
    //#pragma omp distribute parallel for num_threads(NTHREADS_GPU) dist_schedule(static, NTHREADS_GPU)
    for (int j = 0; j < _PB_M; j++){
        mean[j] = 0.0;
        for (int i = 0; i < _PB_N; i++)
            mean[j] += data[i][j];
        mean[j] /= float_n;
    }
}

static void stddev_(int m, int n,
                    DATA_TYPE float_n,
                    DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
                    DATA_TYPE POLYBENCH_1D(mean, M, m),
                    DATA_TYPE POLYBENCH_1D(stddev, M, m))
{
    // #define sqrt_of_array_cell(x, j) sqrt(x[j]);
    DATA_TYPE eps = 0.1f;

    //#pragma omp target teams num_teams((M*N) / NTHREADS_GPU) thread_limit(NTHREADS_GPU) map(tofrom: stddev[0:M]) map(to: mean[0:M], data[0:N][0:M], float_n)
    //#pragma omp distribute parallel for num_threads(NTHREADS_GPU) dist_schedule(static, NTHREADS_GPU)
    for (size_t j = 0; j < _PB_M; j++) {
        stddev[j] = 0.0;
        for (size_t i = 0; i < _PB_N; i++)
            stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
        stddev[j] /= float_n;
        stddev[j] = sqrt_of_array_cell(stddev, j);
        /* The following in an inelegant but usual way to handle
        near-zero std. dev. values, which below would cause a zero-
        divide. */
        stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    }
}

static void center_reduce_(int m, int n,
                           DATA_TYPE float_n,
                           DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
                           DATA_TYPE POLYBENCH_1D(mean, M, m),
                           DATA_TYPE POLYBENCH_1D(stddev, M, m))

{
    //#pragma omp target teams num_teams((M*N) / NTHREADS_GPU) thread_limit(NTHREADS_GPU) map(tofrom: data[0:N][0:M]) map(to: mean[0:M], stddev[0:M], float_n)
    //#pragma omp distribute parallel for num_threads(NTHREADS_GPU) dist_schedule(static, NTHREADS_GPU)
    for (size_t i = 0; i < _PB_N; i++)
        for (size_t j = 0; j < _PB_M; j++){
        data[i][j] -= mean[j];
        data[i][j] /= sqrt(float_n) * stddev[j];
    }
}

static void compute_corr_(int m, int n,
                          DATA_TYPE float_n,
                          DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
                          DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m))
{
  size_t j1, j2, i;
  /* Calculate the m * m correlation matrix. */
  for (j1 = 0; j1 < _PB_M - 1; j1++)
  {
    symmat[j1][j1] = 1.0;
    for (j2 = j1 + 1; j2 < _PB_M; j2++)
    {
      symmat[j1][j2] = 0.0;
      for (i = 0; i < _PB_N; i++)
        symmat[j1][j2] += (data[i][j1] * data[i][j2]);
      symmat[j2][j1] = symmat[j1][j2];
    }
  }
}
  
static void compute_corr_loop_interchange_device_opt_(int m, int n,
                          DATA_TYPE float_n,
                          DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
                          DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m)) {
    /*
    #pragma omp target teams num_teams((_PB_M) / NTHREADS_GPU) thread_limit(NTHREADS_GPU) map(to: data[0:N][0:M]) map(tofrom: symmat[0:M][0:M])
    #pragma omp distribute parallel for num_threads(NTHREADS_GPU) dist_schedule(static, NTHREADS_GPU)
    */


    for (int i = 0; i < _PB_M; i++) {
      symmat[i][i] = 1.0;
      for (int j=i+1; j < _PB_M; j++) {
        symmat[i][j] = 0.0;
      }
    }

    #define NTHREADS_GPU 1024

    #pragma omp target data map(to: data[0:_PB_N][0:_PB_M]) map(tofrom: symmat[0:_PB_M][0:_PB_M])
    for (int i = 0; i < _PB_N; i++) {
      #pragma omp target teams num_teams((_PB_M) / NTHREADS_GPU) thread_limit(NTHREADS_GPU)
       #pragma omp distribute parallel for num_threads(NTHREADS_GPU) dist_schedule(static, NTHREADS_GPU) schedule(static, 1)
      for (int r = 0; r < _PB_N; r++) {
        #pragma omp simd
        for (int c = r+1; c < _PB_M; c++) {
          symmat[r][c] += data[i][r] * data[i][c];
        }
      }
    }

    for (int i = 0; i < _PB_N; i++) {
      for (int j = 0; j < _PB_M; j++) {
        symmat[j][i] = symmat[i][j];
      }
    }

    symmat[_PB_M - 1][_PB_M - 1] = 1.0;
}

static void compute_corr_loop_interchange_not_optimized_(int m, int n,
                                                         DATA_TYPE float_n,
                                                         DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
                                                         DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m))
{
  for (size_t j1 = 0; j1 < _PB_M - 1; j1++)
    for (size_t j2 = j1 + 1; j2 < _PB_M; j2++)
      symmat[j1][j2] = 0.0;

  #if INNERMOST_LOOP_PROFILING_TOGGLE == INNERMOST_LOOP_PROFILING_ENABLE
  puts("PROFILING INNERMOST LOOP EXEC TIMES EVOLUTION:");
  puts("INNERMOST_IT_VS_TIME_US = {");
  #endif

  for (size_t i = 0; i < _PB_N; i++)
    for (size_t j1 = 0; j1 < _PB_M - 1; j1++) {
      symmat[j1][j1] = 1.0;
      
      #if INNERMOST_LOOP_PROFILING_TOGGLE == INNERMOST_LOOP_PROFILING_ENABLE
      if (i > 0) break;
      double t_start = rtclock();
      #endif

      for (size_t j2 = j1 + 1; j2 < _PB_M; j2++)
        symmat[j1][j2] += (data[i][j1] * data[i][j2]);

      #if INNERMOST_LOOP_PROFILING_TOGGLE == INNERMOST_LOOP_PROFILING_ENABLE
      printf("%ld: %lf,\n", j1, (rtclock() - t_start)*1e3);
      #endif
    }

  #if INNERMOST_LOOP_PROFILING_TOGGLE == INNERMOST_LOOP_PROFILING_ENABLE
  puts("}");
  #endif

  for (size_t j1 = 0; j1 < _PB_M - 1; j1++)
    for (size_t j2 = j1 + 1; j2 < _PB_M; j2++)
      symmat[j2][j1] = symmat[j1][j2];
  symmat[_PB_M - 1][_PB_M - 1] = 1.0;
}

static void compute_corr_loop_interchange_task_opt_(int m, int n,
                                                    DATA_TYPE float_n,
                                                    DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
                                                    DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m))
{

  size_t i, j1, j2;

#pragma omp task
  for (i = 0; i < _PB_N; i++)
    symmat[i][i] = 1.0;

  for (j1 = 0; j1 < _PB_M - 1; j1++)
#pragma omp task
#pragma omp simd
    for (j2 = j1 + 1; j2 < _PB_M; j2++)
      symmat[j1][j2] = 0.0;

  int unroll_size_ = 4;
  int blocks = _PB_N / unroll_size_;
#pragma omp taskwait

  for (size_t i = 0; i < blocks; i += 1)
#pragma omp task
    for (j1 = 0; j1 < _PB_M - 1; j1++)
#pragma omp simd
      for (j2 = j1 + 1; j2 < _PB_M; j2++)
      {
        size_t idx = i * unroll_size_;
        symmat[j1][j2] += (data[idx][j1] * data[idx][j2]);
        symmat[j1][j2] += (data[idx + 1][j1] * data[idx + 1][j2]);
        symmat[j1][j2] += (data[idx + 2][j1] * data[idx + 2][j2]);
        symmat[j1][j2] += (data[idx + 3][j1] * data[idx + 3][j2]);
      }
#pragma omp taskwait

  for (size_t i = unroll_size_ * blocks; i < _PB_N; i++)
#pragma omp task
    for (size_t j1 = 0; j1 < _PB_M - 1; j1++)
#pragma omp simd
      for (size_t j2 = j1 + 1; j2 < _PB_M; j2++)
        symmat[j1][j2] += (data[i][j1] * data[i][j2]);
#pragma omp taskwait

  for (size_t j1 = 0; j1 < _PB_M - 1; j1++)
#pragma omp task
#pragma omp simd
    for (size_t j2 = j1 + 1; j2 < _PB_M; j2++)
      symmat[j2][j1] = symmat[j1][j2];

  symmat[_PB_M - 1][_PB_M - 1] = 1.0;
}

static void compute_corr_loop_interchange_parallel_opt_(int m, int n,
                                                        DATA_TYPE float_n,
                                                        DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
                                                        DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m))
{
#pragma omp parallel for
  for (size_t j1 = 0; j1 < _PB_M - 1; j1++)
  {
    symmat[j1][j1] = 1.0;
    for (size_t j2 = j1 + 1; j2 < _PB_M; j2++)
      symmat[j1][j2] = 0.0;
  }

  int unroll_size_ = 4;
  int blocks = _PB_N / unroll_size_;

  #if INNERMOST_LOOP_PROFILING_TOGGLE == INNERMOST_LOOP_PROFILING_ENABLE
  puts("PROFILING INNERMOST LOOP EXEC TIMES EVOLUTION:");
  puts("INNERMOST_IT_VS_TIME_US = {");
  #endif

  for (size_t i = 0; i < blocks; i += 1) {
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t j1 = 0; j1 < _PB_M - 1; j1++) {

      #if INNERMOST_LOOP_PROFILING_TOGGLE == INNERMOST_LOOP_PROFILING_ENABLE
      if (i > 0) continue;
      double t_start = rtclock();
      #endif

      #pragma omp simd
      for (size_t j2 = j1 + 1; j2 < _PB_M; j2++) {
        size_t idx = i * unroll_size_;
        symmat[j1][j2] += (data[idx][j1] * data[idx][j2]);
        symmat[j1][j2] += (data[idx + 1][j1] * data[idx + 1][j2]);
        symmat[j1][j2] += (data[idx + 2][j1] * data[idx + 2][j2]);
        symmat[j1][j2] += (data[idx + 3][j1] * data[idx + 3][j2]);
      }

      #if INNERMOST_LOOP_PROFILING_TOGGLE == INNERMOST_LOOP_PROFILING_ENABLE
      printf("%ld: %lf,\n", j1, (rtclock() - t_start)*1e3);
      #endif
    }
  }

  #if INNERMOST_LOOP_PROFILING_TOGGLE == INNERMOST_LOOP_PROFILING_ENABLE
  puts("}");
  #endif

  for (size_t i = unroll_size_ * blocks; i < _PB_N; i++)
    for (size_t j1 = 0; j1 < _PB_M - 1; j1++)
      for (size_t j2 = j1 + 1; j2 < _PB_M; j2++)
        symmat[j1][j2] += (data[i][j1] * data[i][j2]);

#pragma omp parallel for
  for (size_t j1 = 0; j1 < _PB_M - 1; j1++)
#pragma omp simd
    for (size_t j2 = j1 + 1; j2 < _PB_M; j2++)
      symmat[j2][j1] = symmat[j1][j2];
      
  symmat[_PB_M - 1][_PB_M - 1] = 1.0;
}

static void kernel_correlation_edited(int m, int n,
                                      DATA_TYPE float_n,
                                      DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
                                      DATA_TYPE POLYBENCH_2D(symmat, M, M, m, m),
                                      DATA_TYPE POLYBENCH_1D(mean, M, m),
                                      DATA_TYPE POLYBENCH_1D(stddev, M, m))
{
  int i, j, j1, j2;

  DATA_TYPE eps = 0.1f;

#define sqrt_of_array_cell(x, j) sqrt(x[j])
  polybench_timer_start();
  mean_(m, n, float_n, data, mean);
  polybench_timer_stop();
  printf("Elapsed time for computing mean:");
  polybench_timer_print();

  polybench_timer_start();
  stddev_(m, n, float_n, data, mean, stddev);
  polybench_timer_stop();
  printf("Elapsed time for computing standard deviation:");
  polybench_timer_print();

  polybench_timer_start();
  center_reduce_(m, n, float_n, data, mean, stddev);
  polybench_timer_stop();
  printf("Elapsed time for computing center&reduce:");
  polybench_timer_print();

  polybench_timer_start();
#ifdef NO_OPT
  compute_corr_(m, n, float_n, data, symmat);
#endif
#ifdef LOOP_OPT
  compute_corr_loop_interchange_not_optimized_(m, n, float_n, data, symmat);
#endif
#ifdef TASK_OPT
  compute_corr_loop_interchange_task_opt_(m, n, float_n, data, symmat);
#endif
#ifdef PARALLEL_OPT
  compute_corr_loop_interchange_parallel_opt_(m, n, float_n, data, symmat);
#endif
#ifdef DEVICE_OPT
  compute_corr_loop_interchange_device_opt_(m, n, float_n, data, symmat);
#endif
  polybench_timer_stop();
  printf("Elapsed time for computing correlation:");
  polybench_timer_print();
}

int main(int argc, char **argv)
{

  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data, DATA_TYPE, M, N, m, n);
  POLYBENCH_2D_ARRAY_DECL(symmat_default, DATA_TYPE, M, M, m, m);
  POLYBENCH_2D_ARRAY_DECL(symmat, DATA_TYPE, M, M, m, m);
  POLYBENCH_1D_ARRAY_DECL(mean, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(stddev, DATA_TYPE, M, m);

#ifdef BASELINE
  /* Initialize array(s). */
  init_array(m, n, &float_n, POLYBENCH_ARRAY(data));
  // print_array(m, (data));
  
  /* Start timer. */
  polybench_start_instruments;
  kernel_correlation(m, n, float_n,
                     POLYBENCH_ARRAY(data),
                     POLYBENCH_ARRAY(symmat_default),
                     POLYBENCH_ARRAY(mean),
                     POLYBENCH_ARRAY(stddev));
  polybench_stop_instruments;
  puts("Baseline time [s]: ");
  polybench_print_instruments;
  hash_(POLYBENCH_ARRAY(symmat_default));
#endif

  init_array(m, n, &float_n, POLYBENCH_ARRAY(data));
  polybench_start_instruments;
  kernel_correlation_edited(m, n, float_n,
                            POLYBENCH_ARRAY(data),
                            POLYBENCH_ARRAY(symmat),
                            POLYBENCH_ARRAY(mean),
                            POLYBENCH_ARRAY(stddev));
  polybench_stop_instruments;
  puts("Optimized time [s]: ");
  polybench_print_instruments;
  hash_(POLYBENCH_ARRAY(symmat));
  // print_array(m, POLYBENCH_ARRAY(symmat));
  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat)));
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat_default)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(symmat);
  POLYBENCH_FREE_ARRAY(symmat_default);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}
