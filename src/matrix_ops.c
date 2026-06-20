#include <stdio.h>
#include <math.h>

// For GGUF weights (stored transposed): dst = A @ B^T
int mat_mult_bt(float *dst, const float *A, const float *B, const float *bias,
                size_t rowsA, size_t colsA, size_t rowsB)
{
    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < rowsB; j++) {
            float sum = bias ? bias[j] : 0.f;
            for (size_t k = 0; k < colsA; k++)
                sum += A[i * colsA + k] * B[j * colsA + k];
            dst[i * rowsB + j] = sum;
        }
    }
    return 1;
}

// Pour les activations (row-major normal) : dst = A @ B
int mat_mult(float *dst, const float *A, const float *B, const float *bias,
             size_t rowsA, size_t colsA, size_t colsB)
{
    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            float sum = bias ? bias[j] : 0.f;
            for (size_t k = 0; k < colsA; k++)
                sum += A[i * colsA + k] * B[k * colsB + j];
            dst[i * colsB + j] = sum;
        }
    }
    return 1;
}

void transp(float * restrict dst, const float * restrict A, size_t rows, size_t cols)
{

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            dst[j * rows + i] = A[i * cols + j];
        }
    }

}

void softmax_causal(float *X, size_t q_len, size_t kv_len, size_t pos)
{
    for (size_t i = 0; i < q_len; i++) {
        float *row = X + i * kv_len;
        size_t visible = pos + i + 1;

        float max = row[0];
        for (size_t j = 1; j < visible; j++)
            if (row[j] > max) max = row[j];

        float sum = 0.f;
        for (size_t j = 0; j < visible; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        for (size_t j = 0; j < visible; j++)
            row[j] /= sum;

        for (size_t j = visible; j < kv_len; j++)
            row[j] = 0.f;
    }
}