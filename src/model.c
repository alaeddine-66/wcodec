#include <stdint.h>
#include <string.h>

#include "../include/header.h"
#include "../include/quants.h"
#include "../include/model.h"

float *embedding_forward(
    FILE *fp, 
    gguf_tensor_info_t *tensor,
    const uint32_t *tokens,
    Model *m,
    long tensor_data_offset,
    Arena *arena
){
    if (tensor->n_dimensions != 2) return NULL;
    m->d_model = tensor->dimensions[0];
    if (m->d_model % QK_K != 0) return NULL;

    float *out = arena_alloc(arena, m->seq_len * m->d_model * sizeof(float));
    if (!out) return NULL;

    uint32_t nb_blocks = m->d_model / QK_K;

    long start =
        tensor_data_offset +
        tensor->offset;

    for (uint32_t i = 0; i < m->seq_len; i++) {
        float *dst = &out[i * m->d_model];
        if (tensor->type == GGML_TYPE_Q6_K){
            long pos = start + (long)tokens[i] * nb_blocks * sizeof(block_q6_K);
            if (fseek(fp, pos, SEEK_SET)) return NULL;
            block_q6_K blocks[nb_blocks];
            if (fread(blocks, sizeof(block_q6_K), nb_blocks, fp) != nb_blocks) return NULL;
            dequantize_q6_K(blocks, dst, nb_blocks);
        }else{
            fprintf(stderr, "Error: unknown value type %u\n", tensor->type);
            return NULL;
        }
    }

    return out;
}

