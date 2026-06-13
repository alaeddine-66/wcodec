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

static inline void get_scale_min_k4(int j, const uint8_t *sc, uint8_t *scale, uint8_t *min)
{
    if (j < 4) {
        *scale = sc[j]     & 0x3F;
        *min   = sc[j + 4] & 0x3F;
    } else {
        *scale = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
        *min   = (sc[j + 4] >> 4)   | ((sc[j]     >> 6) << 4);
    }
}

void dequantize_q4_K(block_q4_K *blocks, float *src, uint32_t nb_blocks)
{
    for (uint32_t i = 0; i < nb_blocks; i++) {
        const block_q4_K *b = &blocks[i];
        const float d    = fp16_to_fp32(b->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        const float dmin = fp16_to_fp32(b->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);

        const uint8_t *q  = b->qs;
        const uint8_t *sc = b->scales;

        float *out = src + i * QK_K;
        int is = 0;

        for (int j = 0; j < QK_K; j += 64) {
            uint8_t scale0, min0, scale1, min1;
            get_scale_min_k4(is + 0, sc, &scale0, &min0);
            get_scale_min_k4(is + 1, sc, &scale1, &min1);

            float d1 = d * scale0, m1 = dmin * min0;
            float d2 = d * scale1, m2 = dmin * min1;

            for (int l = 0; l < 32; l++) out[l]      = d1 * (q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; l++) out[l + 32] = d2 * (q[l] >> 4)   - m2;

            out += 64;
            q   += 32;
            is  += 2;
        }
    }
}

