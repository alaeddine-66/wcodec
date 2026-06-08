#include "unity.h"
#include "../include/quants.h"
#include "../include/header.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void setUp(void)    {}
void tearDown(void) {}

#define FP16_EPS 1e-3f
#define K_SCALE_SIZE 16
#define DEQUANT_EPS 1e-4f

#define TEST_ASSERT_FLOAT_APPROX(expected, got, eps) \
    TEST_ASSERT_FLOAT_WITHIN((eps) * (fabsf(expected) > 1.f ? fabsf(expected) : 1.f), \
                             (expected), (got))

/* ------------------------------------------------------------------
 * Reference implementation from GGML / llama.cpp.
 * Source: ggml-common.h ggml-quants.h
 *
 * Used as a validation baseline for testing my own implementation.
 * ------------------------------------------------------------------ */

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static uint16_t ggml_compute_fp32_to_fp16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

static inline float ggml_compute_fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

/* ================================================================== */
/*  dequantize_q6_K — Reference (ggml-quants.c, llama.cpp)            */
/* ================================================================== */
static void __dequantize_q6_K(const block_q6_K *b, float *dst)
{
    const float    d  = fp16_to_fp32(b->d);
    const uint8_t *ql = b->ql;
    const uint8_t *qh = b->qh;
    const int8_t  *sc = b->scales;

    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            const int is = l / 16;

            const int8_t q1 = (int8_t)(((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            const int8_t q2 = (int8_t)(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            const int8_t q3 = (int8_t)(((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            const int8_t q4 = (int8_t)(((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32);

            dst[n + l +  0] = d * sc[is + 0] * q1;
            dst[n + l + 32] = d * sc[is + 2] * q2;
            dst[n + l + 64] = d * sc[is + 4] * q3;
            dst[n + l + 96] = d * sc[is + 6] * q4;
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

void dequantize_q6_K_ref(const block_q6_K *blocks,
                           float *dst, uint32_t nb_blocks)
{
    for (uint32_t i = 0; i < nb_blocks; i++)
        __dequantize_q6_K(&blocks[i], dst + i * QK_K);
}

block_q6_K make_test_block(float delta, const int8_t scales[K_SCALE_SIZE])
{
    block_q6_K b;
    memset(&b, 0, sizeof b);
    b.d = ggml_compute_fp32_to_fp16(delta);
    memcpy(b.scales, scales, K_SCALE_SIZE);

    /*
     * Inverse de dequantize_q6_K_ref :
     * Pour chaque groupe de 128, pour l ∈ [0,32) on encode
     * 4 valeurs v1..v4 ∈ [0,63] (avant centrage).
     */
    uint8_t *qlw = b.ql, *qhw = b.qh;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            uint8_t v1 = (uint8_t)((n + l +  0) % 64);
            uint8_t v2 = (uint8_t)((n + l + 32) % 64);
            uint8_t v3 = (uint8_t)((n + l + 64) % 64);
            uint8_t v4 = (uint8_t)((n + l + 96) % 64);

            qlw[l     ] = (v1 & 0xFu) | (uint8_t)((v3 & 0xFu) << 4);
            qlw[l + 32] = (v2 & 0xFu) | (uint8_t)((v4 & 0xFu) << 4);
            qhw[l]      = (uint8_t)(
                            ((v1 >> 4) & 3u)        |
                            (((v2 >> 4) & 3u) << 2) |
                            (((v3 >> 4) & 3u) << 4) |
                            (((v4 >> 4) & 3u) << 6));
        }
        qlw += 64;
        qhw += 32;
    }
    return b;
}

void test_fp16_max_normal(void)
{
    TEST_ASSERT_FLOAT_APPROX(65504.f, fp16_to_fp32(ggml_compute_fp32_to_fp16(65504.f)), FP16_EPS);
}

void test_fp16_min_normal(void)
{
    TEST_ASSERT_FLOAT_APPROX(6.10352e-5f, fp16_to_fp32(ggml_compute_fp32_to_fp16(6.10352e-5f)), FP16_EPS);
}

void test_fp16_min_denorm(void)
{
    TEST_ASSERT_FLOAT_APPROX(5.96046e-8f, fp16_to_fp32(ggml_compute_fp32_to_fp16(5.96046e-8f)), FP16_EPS);
}

void test_fp16_inf_positive(void)
{
    float v = fp16_to_fp32(0x7C00u);
    TEST_ASSERT_TRUE(isinf(v) && v > 0.f);
}

void test_fp16_inf_negative(void)
{
    float v = fp16_to_fp32(0xFC00u);
    TEST_ASSERT_TRUE(isinf(v) && v < 0.f);
}

void test_fp16_roundtrip_typical_values(void)
{
    float vals[] = { 0.f, -0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f,
                     -0.25f, -1.f, 100.f, 0.001f };
    for (size_t i = 0; i < sizeof vals / sizeof *vals; i++) {
        float recovered = fp16_to_fp32(ggml_compute_fp32_to_fp16(vals[i]));
        TEST_ASSERT_FLOAT_APPROX(vals[i], recovered, FP16_EPS);
    }
}

void test_fp16_sweep_all_non_nan(void)
{
    /*
     * Sweep de toutes les valeurs fp16 non-NaN :
     * vérifie que fp16_to_fp32 et la ref sont d'accord.
     */
    int diffs = 0;
    for (uint32_t h = 0; h <= 0xFFFFu; h++) {
        uint16_t hh = (uint16_t)h;
        /* sauter les NaN (exp=31, mant≠0) */
        if ((hh & 0x7C00u) == 0x7C00u && (hh & 0x03FFu)) continue;

        float got = fp16_to_fp32(hh);
        float ref = ggml_compute_fp16_to_fp32(hh);
        float eps = 1e-6f * (fabsf(ref) + 1e-30f);

        if (fabsf(got - ref) > eps) diffs++;
    }
    TEST_ASSERT_EQUAL_INT(0, diffs);
}

void test_dequant_yours_differs_from_ref(void)
{
    const int NB = 2;
    int8_t sc[K_SCALE_SIZE];
    for(uint32_t i = 0; i < K_SCALE_SIZE; i++)
        sc[i] = 3;
    block_q6_K blocks[2];
    for (int i = 0; i < NB; i++)
        blocks[i] = make_test_block(0.25f, sc);

    float ref[NB * QK_K], got[NB * QK_K];
    dequantize_q6_K_ref(blocks, ref, NB);
    dequantize_q6_K(blocks, got, (uint32_t)NB);

    int diffs = 0;
    for (int i = 0; i < NB * QK_K; i++)
        if (fabsf(ref[i] - got[i]) > DEQUANT_EPS) diffs++;

    TEST_ASSERT_EQUAL_INT(0, diffs);
}

int main(void)
{
    UNITY_BEGIN();

    /* -- fp16_to_fp32 -- */
    RUN_TEST(test_fp16_max_normal);
    RUN_TEST(test_fp16_min_normal);
    RUN_TEST(test_fp16_min_denorm);
    RUN_TEST(test_fp16_inf_positive);
    RUN_TEST(test_fp16_inf_negative);
    RUN_TEST(test_fp16_roundtrip_typical_values);
    RUN_TEST(test_fp16_sweep_all_non_nan);

    /* -- dequantize_q6_K -- */
    RUN_TEST(test_dequant_yours_differs_from_ref);

    return UNITY_END();
}