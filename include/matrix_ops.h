#ifndef MATRIX_OPS
#define MATRIX_OPS

#include <stdio.h>

int mat_mult(float *dst, const float *A, const float *B, const float *bias,
             size_t rowsA, size_t colsA, size_t colsB);

int mat_mult_bt(float *dst, const float *A, const float *B, const float *bias,
                size_t rowsA, size_t colsA, size_t rowsB);

void transp(float * restrict dst, const float * restrict A, size_t rows, size_t cols);

void softmax_causal(float *X, size_t q_len, size_t kv_len, size_t pos);

#endif
