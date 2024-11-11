#include <checksum.h>

void compute_checksums(
  DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
  DATA_TYPE POLYBENCH_1D(out_cols_sum, N, n),
  DATA_TYPE POLYBENCH_1D(out_rows_sum, M, M)
);

bool verify_checksums(
  const DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
);

