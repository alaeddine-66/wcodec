#ifndef ROPE_H
#define ROPE_H

#include <stdio.h>
#include "allocator.h"

typedef struct {
    float *cos;
    float *sin;
} RopeTable;

RopeTable build_rope_table(Arena *arena, size_t seq_len, size_t head_dim, float rope_theta);
void apply_rope(float *X,size_t seq_len, size_t n_heads, size_t head_dim, RopeTable rt);

#endif