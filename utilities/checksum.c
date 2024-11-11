#include <checksum.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(VERIFY_CHECKSUM)
#  define FILENAME_DECL(filename) static const char *FILENAME = filename

#  if defined(MINI_DATASET)
#   define FILENAME_DECL("mini_csum.bin")
#  elif defined(SMALL_DATASET)
#   define FILENAME_DECL("small_csum.bin")
#  elif defined(STANDARD_DATASET)
#   define FILENAME_DECL("standard_csum.bin")
#  elif defined(LARGE_DATASET)
#   define FILENAME_DECL("large_csum.bin")
#  elif defined(EXTRALARGE_DATASET)
#   define FILENAME_DECL("extralarge_csum.bin")
#  else
#   error Checksum verfication enabled without specifying the expected results
#  endif

#endif

DATA_TYPE MACHINE_EPSILON = nextafter (1.0, +2.0) - 1.0;


void compute_checksums(
  DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
  DATA_TYPE POLYBENCH_1D(out_cols_sum, M, M),
  DATA_TYPE POLYBENCH_1D(out_rows_sum, N, n)
) {
  for (i = 0; i < _PB_N; i++) out_rows_sum[i] = 0;
  for (i = 0; i < _PB_M; i++) out_cols_sum[i] = 0;

  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j < _PB_M; j++) {
      out_rows_sum[i] += data[i][j];
      out_cols_sum[j] += data[i][j];
    }
  }
}

void save_checksums(
  const DATA_TYPE POLYBENCH_1D(out_cols_sum, M, M),
  const DATA_TYPE POLYBENCH_1D(out_rows_sum, N, n)
) {
  FILE *file = fopen(FILENAME, "wb");
  unsigned error_num = 1;
  
  if (!file) {
    perror("File not opened");
    exit(error_num++);
  }

  if (fwrite(out_cols_sum, sizeof(DATA_TYPE), POLYBENCH_C99_SELECT(M, m), file) != POLYBENCH_C99_SELECT(M, m)) {
    perror("Erorr while writing columns' checksums");
    exit(error_num++);
  }

  if (fwrite(out_rows_sum, sizeof(DATA_TYPE), POLYBENCH_C99_SELECT(N, n), file) != POLYBENCH_C99_SELECT(N, n)) {
    perror("Erorr while writing columns' checksums");
    exit(error_num++);
  }

  fclose(file);
}

bool verify_checksums(
  const DATA_TYPE POLYBENCH_2D(data, M, N, m, n)
) {
  // Read from file
  POLYBENCH_1D_ARRAY_DECL(expected_csum_cols,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(expected_csum_rows,DATA_TYPE,N,n);

  FILE *file = fopen(FILENAME, "rb");
  unsigned error_num = 1;
  
  if (!file) {
    perror("File not opened");
    exit(error_num++);
  }

  if (fread(expected_csum_cols, sizeof(DATA_TYPE), POLYBENCH_C99_SELECT(M, m), file) != POLYBENCH_C99_SELECT(M, m)) {
    perror("Erorr while writing columns' checksums");
    exit(error_num++);
  }

  if (fwrite(expected_csum_rows, sizeof(DATA_TYPE), POLYBENCH_C99_SELECT(N, n), file) != POLYBENCH_C99_SELECT(N, n)) {
    perror("Erorr while writing columns' checksums");
    exit(error_num++);
  }

  fclose(file);

  // Recompute checksum
  POLYBENCH_1D_ARRAY_DECL(csum_cols,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(csum_rows,DATA_TYPE,N,n);

  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j < _PB_M; j++) {
      csum_rows[i] += data[i][j];
      csum_cols[j] += data[i][j];
    }
  }

  for (i = 0; i < _PB_M; i++) {
    DATA_TYPE delta = abs(csum_cols[i] - expected_csum_cols[i]);
    DATA_TYPE abs_err = max(csum_cols[i], expected_csum_cols[i]) * POLYBENCH_C99_SELECT(M, m) * MACHINE_EPSILON;
    if (delta > abs_err) return false;
  }
  
  for (i = 0; i < _PB_N; i++) {
    DATA_TYPE delta = abs(csum_rows[i] - expected_csum_rows[i]);
    DATA_TYPE abs_err = max(csum_rows[i], expected_csum_rows[i]) * POLYBENCH_C99_SELECT(M, m) * MACHINE_EPSILON;
    if (delta > abs_err) return false;
  }

  return true;
}

