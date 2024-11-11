#ifndef CHECKSUM_H_
#define CHECKSUM_H_

void compute_checksums(
  DATA_TYPE POLYBENCH_2D(data, M, N, m, n),
  DATA_TYPE POLYBENCH_1D(out_cols_sum, M, M),
  DATA_TYPE POLYBENCH_1D(out_rows_sum, N, n)
);

void save_checksums(
  const DATA_TYPE POLYBENCH_1D(out_cols_sum, M, M),
  const DATA_TYPE POLYBENCH_1D(out_rows_sum, N, n)
);

bool verify_checksums(
  const DATA_TYPE POLYBENCH_2D(data, M, N, m, n)
);

#endif  // CHECKSUM_H_
