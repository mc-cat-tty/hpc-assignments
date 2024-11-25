#ifndef __CORRELATION_H__
#define __CORRELATION_H__



# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define STANDARD_DATASET
# endif

# if !defined(N) && !defined(M)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#   define N 32
#   define M 32
#  endif

#  ifdef SMALL_DATASET
#   define N 500
#   define M 500
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
#   define N 1000
#   define M 1000
#  endif

#  ifdef LARGE_DATASET
#   define N 2000
#   define M 2000
#  endif

#  ifdef EXTRALARGE_DATASET
#   define N 4000
#   define M 4000
#  endif
# endif 


# ifndef DATA_TYPE
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif


#endif /*__CORRELATION_H__*/
