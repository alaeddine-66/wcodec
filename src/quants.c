#include "../include/quants.h"

float fp16_to_fp32(uint16_t h)
{
    union {
        uint32_t u;
        float f;
    } out;

    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;

    if (exp == 0) {
        if (man == 0) {
            out.u = sign << 31;
            return out.f;
        }

        // subnormal
        while ((man & 0x400) == 0) {
            man <<= 1;
            exp--;
        }

        man &= 0x3FF;
        exp++;
    }
    else if (exp == 31) {
        // inf / NaN
        out.u =
            (sign << 31) |
            (0xFF << 23) |
            (man << 13);

        return out.f;
    }

    exp = exp + 112;

    out.u =
        (sign << 31) |
        (exp << 23) |
        (man << 13);

    return out.f;
}

void dequantize_q6_K(block_q6_K *blocks, float *src, uint32_t nb_blocks )
{
    for(uint32_t i = 0; i < nb_blocks; i++){
        const block_q6_K *b = &blocks[i];
        const float d = fp16_to_fp32(b->d);
        
        const uint8_t *ql = b->ql;
        const uint8_t *qh = b->qh;
        const int8_t  *sc = b->scales;

        float *out = src + i * QK_K;

        for (int n = 0; n < QK_K; n += 128){
            for(int j = 0; j < 32; j++){
                const int8_t s_j = j/16;
                const int8_t q1 = (((qh[j] & 0x03) << 4)  | (ql[j + 0]  & 0x0F) ) - 32;
                const int8_t q2 = (((qh[j] & 0x0C) << 2)  | (ql[j + 32] & 0x0F) ) - 32;
                const int8_t q3 = ((qh[j] & 0x30)         | (ql[j + 0]  >> 4)   ) - 32;
                const int8_t q4 = (((qh[j] & 0xC0) >> 2)  | (ql[j + 32] >> 4)   ) - 32;
                out[j +  0] = d * sc[s_j + 0] * q1;
                out[j + 32] = d * sc[s_j + 2] * q2;
                out[j + 64] = d * sc[s_j + 4] * q3;
                out[j + 96] = d * sc[s_j + 6] * q4;
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

