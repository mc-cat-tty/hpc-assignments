#include <checksum.h>
#include <stdio.h>
#include <stdlib.h>

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


void compute_checksums(
  DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
  DATA_TYPE POLYBENCH_1D(out_cols_sum, N, n),
  DATA_TYPE POLYBENCH_1D(out_rows_sum, M, M)
) {
}

void save_checksums(
  DATA_TYPE POLYBENCH_1D(out_cols_sum, N, n)
  DATA_TYPE POLYBENCH_1D(out_rows_sum, M, M)
) {
  FILE *file = fopen(FILENAME, "wb");
  unsigned error_num = 1;
  
  if (!file) {
    perror("File not opened");
    exit(error_num++);
  }

  if (fwrite(out_cols_sum, sizeof(DATA_TYPE), POLYBENCH_C99_SELECT(N, n), file) != POLYBENCH_C99_SELECT(N, n)) {
    perror("Erorr while writing columns' checksums");
    exit(error_num++);
  }

  if (fwrite(out_rows_sum, sizeof(DATA_TYPE), POLYBENCH_C99_SELECT(M, m), file) != POLYBENCH_C99_SELECT(M, m)) {
    perror("Erorr while writing columns' checksums");
    exit(error_num++);
  }

  fclose(file);
}

bool verify_checksums(
  const DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
) {

}


