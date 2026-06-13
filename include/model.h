#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include "allocator.h"
#include "header.h"
#include "rope.h"

typedef struct {
    uint32_t seq_len, d_model, head_count_kv, head_count;
}Model;

float *transformer(Arena *arena, FILE *f, gguf_header_t *header, uint32_t *tokens, Model *m, RopeTable rt, size_t vocab_size);

#endif