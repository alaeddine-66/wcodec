#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/rope.h"
#include "../include/allocator.h"

static float *build_rope_freqs(size_t head_dim, float rope_theta)
{
    size_t n_pairs = head_dim / 2;

    float *freq = malloc(n_pairs * sizeof(float));
    if (!freq) {
        fprintf(stderr, "malloc error in build_rope_freqs\n");
        return NULL;
    }

    for (size_t i = 0; i < n_pairs; i++) {
        freq[i] = powf(
            rope_theta,
            -2.0f * (float)i / (float)head_dim
        );
    }

    return freq;
}

RopeTable build_rope_table(
    Arena *arena,
    size_t seq_len,
    size_t head_dim,
    float rope_theta
)
{
    size_t n_pairs = head_dim / 2;

    RopeTable rt;
    rt.cos = arena_alloc(arena, seq_len * n_pairs * sizeof(float));
    rt.sin = arena_alloc(arena, seq_len * n_pairs * sizeof(float));
    if (!rt.cos || !rt.sin) return rt;

    float *freq = build_rope_freqs(head_dim, rope_theta);

    for (size_t t = 0; t < seq_len; t++) {
        for (size_t i = 0; i < n_pairs; i++) {

            float angle = t * freq[i];

            size_t idx = t * n_pairs + i;

            rt.cos[idx] = cosf(angle);
            rt.sin[idx] = sinf(angle);
        }
    }

    free(freq);
    return rt;
}

void apply_rope( float *X,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    RopeTable rt
){
    size_t dim = n_heads * head_dim;
    size_t n_pairs = head_dim / 2;

    for (size_t t = 0; t < seq_len; t++) {
        float *xt = X + t * dim;
        for (size_t h = 0; h < n_heads; h++) {
            float *xh = xt + h * head_dim;
            for (size_t i = 0; i < n_pairs; i++) {

                size_t idx = t * n_pairs + i;

                float c = rt.cos[idx];
                float s = rt.sin[idx];

                float x0 = xh[2*i];
                float x1 = xh[2*i + 1];

                float ksum = c * (x0 + x1);

                xh[2*i]     = ksum - (c + s) * x1;
                xh[2*i+1]   = ksum + (s - c) * x0;
            }
        }
    }
}