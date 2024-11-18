# Group 2 Assignments for HPC course

## Assignments
The assignments involve parallelizing `correlation` with different frameworks:
1. Assignment 1 involves using OpenMP v4.x.
2. Assignment 2 involves using CUDA v10.0.
3. Assignment 3 TBD

## Usage

### Parameter Configuration
Parameters such as the input sizes, data type, and threshold for GPU-CPU output comparisons can be enabled/disabled or modified in `options.mk` file.

After modifying, run `make clean` then `make` on relevant code for modifications to take effect in resulting executable.


#### Input Size
By default the `STANDARD_DATASET` as defined in the `<benchmark>.h` file is used as the input size.  The dataset choice can be adjusted from `STANDARD_DATASET` to other options (`MINI_DATASET`, `SMALL_DATASET`, etc) in the `options.mk` file.

#### `DATA_TYPE`
By default, the `DATA_TYPE` used in these codes are `float` that can be changed to `double` by changing the `DATA_TYPE` typedef. Note that in OpenCL, the `DATA_TYPE` needs to be changed in both the .h and .cl files, as the .cl files contain the kernel code and is compiled separately at run-time.


#### Other available options

These are passed as macro definitions during compilation time 
(e.g `-Dname_of_the_option`) or can be added with a `#define` to the code.
- `POLYBENCH_STACK_ARRAYS` (only applies to allocation on host): 
use stack allocation instead of malloc [default: off]
- `POLYBENCH_DUMP_ARRAYS`: dump all live-out arrays on stderr [default: off]
- `POLYBENCH_CYCLE_ACCURATE_TIMER`: Use Time Stamp Counter to monitor
  the execution time of the kernel [default: off]
- `MINI_DATASET`, `SMALL_DATASET`, `STANDARD_DATASET`, `LARGE_DATASET`,
  `EXTRALARGE_DATASET`: set the dataset size to be used
  [default: `STANDARD_DATASET`]

- `POLYBENCH_USE_C99_PROTO`: Use standard C99 prototype for the functions.

- `POLYBENCH_USE_SCALAR_LB`: Use scalar loop bounds instead of parametric ones.
