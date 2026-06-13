#ifndef MATRIX_OPS
#define MATRIX_OPS

#include <stdio.h>

int mat_mult(float * restrict dst, const float * restrict A, const float * restrict B, const float * restrict bias,
                size_t rowsA, size_t colsA, size_t rowsB, size_t colsB);

void transp(float * restrict dst, const float * restrict A, size_t rows, size_t cols);

void softmax_causal(float *X, size_t n);

#endif
