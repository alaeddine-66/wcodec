#ifndef QUANTS_H
#define QUANTS_H

#include <stdint.h>
#include <stdio.h>

#define QK_K 256
#define K_SCALE_SIZE_Q4 12


// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    union {
        struct {
            uint16_t d;    // super-block scale for quantized scales
            uint16_t dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        uint32_t dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE_Q4]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;

// 6-bit quantization
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    uint16_t d;             // super-block scale
} block_q6_K;

void dequantize_q6_K(block_q6_K *block, float *src, uint32_t nb_blocks);
void dequantize_q4_K(block_q4_K *blocks, float *src, uint32_t nb_blocks);
float fp16_to_fp32(uint16_t h);

#endif