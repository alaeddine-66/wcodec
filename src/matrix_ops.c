#include <stdio.h>
#include <math.h>

int mat_mult(float * restrict dst, const float * restrict A, const float * restrict B, const float * restrict bias,
                size_t rowsA, size_t colsA, size_t rowsB, size_t colsB)
{
    if (!A || !B) {
        fprintf(stderr, "Error, trying to multiply NULL matrix\n");
        return 0;
    }
    if (colsA != rowsB){
        fprintf(stderr, "Error in mat_mult, colsA and rowsB should be the same\n");
        return 0;
    }

    for (size_t i = 0; i < rowsA; i++) {
        for (size_t j = 0; j < colsB; j++) {
            float sum = (bias ? bias[j] : 0.0f);

            for (size_t k = 0; k < colsA; k++) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }

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

void softmax_causal(float *X, size_t n)
{
    for(size_t i = 0; i < n; i++){
        float max = X[i * n];
        for(size_t j = 1; j <= i; j++){
            if (X[i * n + j] > max) max = X[i * n + j];
        }
        float sum = 0.f;
        for(size_t j = 0; j <= i; j++){
            sum += expf(X[i * n + j] - max);
        }

        for(size_t j = 0; j <= i; j++){
            X[i * n + j] = expf(X[i * n + j] - max) / sum;
        }
    }

    for(size_t i = 0; i < n; i++)
        for(size_t j = i + 1; j < n; j++)
            X[i * n + j] = 0;

}