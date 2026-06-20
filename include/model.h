#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include "allocator.h"
#include "header.h"
#include "rope.h"

typedef struct {
    uint32_t seq_len, d_model, head_count_kv, head_count, block_count;
    float **K_cache, **V_cache;
}Model;

float *load_matrix(gguf_tensor_info_t *mat, FILE *fp, long start_offset);

float *embedding_forward(FILE *fp, gguf_tensor_info_t *tensor, const uint32_t *tokens,
    Model *m, long tensor_data_offset, Arena *arena
);

void RMSNorm(Model *m, float *out, const float *g, float eps);
void pack_head(float * restrict dst, const float * restrict X, size_t seq_len, size_t d_model,
                 size_t head_dim, size_t head_idx);

int attention_forward(FILE *f, Model *m, long start_offset, size_t layer,
            gguf_tensor_info_t *K_w,
            gguf_tensor_info_t *V_w,
            gguf_tensor_info_t *Q_w,
            gguf_tensor_info_t *K_b,
            gguf_tensor_info_t *V_b,
            gguf_tensor_info_t *Q_b,
            gguf_tensor_info_t *Wo,
            gguf_tensor_info_t *Wo_b,
            float *x, size_t pos, float *out, RopeTable rt);

float *feed_forward(FILE *f, Model *m,long start_offset,
            gguf_tensor_info_t *W_up,
            gguf_tensor_info_t *W_gate,
            gguf_tensor_info_t *W_down,
            float *x);

float *transformer(Arena *arena, FILE *f, gguf_header_t *header, uint32_t *tokens, Model *m, RopeTable rt, size_t vocab_size, size_t pos);

#endif