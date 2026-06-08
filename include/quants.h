#ifndef QUANTS_H
#define QUANTS_H

#include <stdint.h>
#include <stdio.h>

#define QK_K 256

// 6-bit quantization
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    uint16_t d;             // super-block scale
} block_q6_K;

void dequantize_q6_K(block_q6_K *block, float *src, uint32_t nb_blocks);
float fp16_to_fp32(uint16_t h);

#endif