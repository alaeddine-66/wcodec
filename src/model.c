#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "../include/header.h"
#include "../include/quants.h"
#include "../include/model.h"
#include "../include/rope.h"
#include "../include/matrix_ops.h"

float *load_matrix(gguf_tensor_info_t *mat, FILE *fp, long start_offset);

float *embedding_forward(
    FILE *fp,
    gguf_tensor_info_t *tensor,
    const uint32_t *tokens,
    Model *m,
    long tensor_data_offset,
    Arena *arena
){
    if (!tensor || tensor->n_dimensions != 2) {
        fprintf(stderr, "Error, embed shape should be d_model * vocab_size\n");
        return NULL;
    }

    m->d_model = tensor->dimensions[0];
    const size_t vocab_size = tensor->dimensions[1];

    if (m->d_model % QK_K != 0) {
        fprintf(stderr, "Error, d_model should be multiple of %d\n", QK_K);
        return NULL;
    }

    float *W = load_matrix(tensor, fp, tensor_data_offset);
    if (!W) return NULL;

    float *out = arena_alloc(arena, m->seq_len * m->d_model * sizeof(float));
    if (!out) {
        free(W);
        return NULL;
    }

    for (uint32_t i = 0; i < m->seq_len; i++) {
        uint32_t tok = tokens[i];
        if (tok >= vocab_size) {
            fprintf(stderr, "Error, token %u out of vocab range (%zu)\n", tok, vocab_size);
            free(W);
            return NULL;
        }

        float *dst = &out[i * m->d_model];

        for (size_t k = 0; k < m->d_model; k++) {
            dst[k] = W[k * vocab_size + tok];
        }
    }

    free(W);
    return out;
}

void RMSNorm(Model *m, float *out, const float *g, float eps)
{

    for(size_t t = 0; t < m->seq_len; t++){
        float *x = out + t * m->d_model;

        float rms = 0.f;
        for(size_t i = 0; i < m->d_model; i++){
            rms += x[i] * x[i];
        } 
        rms = sqrtf(rms / m->d_model + eps);
        for(size_t i = 0; i < m->d_model; i++){
            x[i] = (x[i] / rms) * g[i];
        }
    }
}

float *load_matrix(gguf_tensor_info_t *mat, FILE *fp, long start_offset)
{
    if (!mat) {
        fprintf(stderr, "Error, trying to load NULL matrix\n");
        return NULL;
    }

    long pos = start_offset + mat->offset;
    if (fseek(fp, pos, SEEK_SET)){
        fprintf(stderr, "fseek error in load_matrix for %s", mat->name.string);
        return NULL;
    }
    size_t size = 1;
    for(uint32_t i = 0; i < mat->n_dimensions; i++)
        size *= mat->dimensions[i];
    
    if (size % QK_K != 0) {
        fprintf(stderr, "Error, %s dimension should be multiple of %d ", mat->name.string, QK_K);
        return 0;
    }
    size_t nb_blocks = size / QK_K;

    float *res = malloc(size * sizeof(float));
    if (!res){
        fprintf(stderr, "Error, malloc in load_matrix\n");
        return NULL;
    }

    switch (mat->type){
        case GGML_TYPE_F32:{
            if (fread(res, sizeof(float), size, fp) != size){
                free(res); fprintf(stderr, "fread error in %s", mat->name.string); return NULL;
            }
            break; 
        }
        case GGML_TYPE_Q6_K :{
            block_q6_K *blocks = malloc(nb_blocks * sizeof(block_q6_K));
            if (fread(blocks, sizeof(block_q6_K), nb_blocks, fp) != nb_blocks){
                free(blocks); free(res); fprintf(stderr, "fread error in %s", mat->name.string); return NULL;
            } 
            dequantize_q6_K(blocks, res, nb_blocks);
            free(blocks);
            break;
        }
        case GGML_TYPE_Q4_K:{
            block_q4_K *blocks = malloc(nb_blocks * sizeof(block_q4_K));
            if (fread(blocks, sizeof(block_q4_K), nb_blocks, fp) != nb_blocks){
                free(blocks); free(res); fprintf(stderr, "fread error in %s", mat->name.string); return NULL;
            };
            dequantize_q4_K(blocks, res, nb_blocks);
            free(blocks);
            break;
        }
        default:
            free(res);
            printf("Error, Not supported format for %s\n", mat->name.string);
            return NULL;
    }
    return res;
}

static void pack_head(float * restrict dst, const float * restrict X, size_t seq_len, size_t d_model,
                 size_t head_dim, size_t head_idx)
{
    for (size_t t = 0; t < seq_len; t++) {
        memcpy(
            dst + t * head_dim,
            X + t * d_model + head_idx * head_dim,
            head_dim * sizeof(float)
        );
    }

}

static int projection(FILE *f, Model *m, long start_offset, float *restrict out, 
        const float *restrict x, gguf_tensor_info_t *W, gguf_tensor_info_t *b,
        size_t row_mat, size_t col_mat
){
    float *mat = load_matrix(W, f, start_offset);
    if (!mat) {return 0;}
    float *bias = load_matrix(b, f, start_offset);
    if (!bias) {free(mat); return 0;}
    int ok = mat_mult(out, x, mat, bias, m->seq_len, m->d_model , row_mat, col_mat);
    free(mat); free(bias);
    return ok;
}

int attention_forward(FILE *f, Model *m, long start_offset,
            gguf_tensor_info_t *K_w,
            gguf_tensor_info_t *V_w,
            gguf_tensor_info_t *Q_w,
            gguf_tensor_info_t *K_b,
            gguf_tensor_info_t *V_b,
            gguf_tensor_info_t *Q_b,
            gguf_tensor_info_t *Wo,
            gguf_tensor_info_t *Wo_b,
            float *x, float *out, RopeTable rt)
{

    if (m->head_count == 0 || m->head_count_kv == 0) {
        fprintf(stderr, "head_count and head_count_kv should not be equal to zero\n");
        return 0;
    }

    if (m->head_count % m->head_count_kv != 0) {
        fprintf(stderr, "head_count should be a multiple of head_count_kv\n");
        return 0;
    }

    if (m->d_model % m->head_count != 0) {
        fprintf(stderr, "d_model and head_count should not be equal to zero\n");
        return 0;
    }

    uint32_t head_dim   = m->d_model / m->head_count;
    uint32_t group_size = m->head_count / m->head_count_kv;
    uint32_t kv_width   = m->head_count_kv * head_dim;
    
    Arena arena = {0};
    size_t buf_Q_size = m->seq_len * m->d_model;
    size_t buf_KV_size  = m->seq_len * kv_width;
    float *proj = arena_alloc(&arena, (buf_Q_size + 2 * buf_KV_size) * sizeof(float));
    if (!proj) return 0; 

    float *Q_x = proj;
    float *K_x = Q_x + buf_Q_size;
    float *V_x = K_x + buf_KV_size;

    if (!projection(f, m, start_offset, Q_x, x, Q_w, Q_b, m->d_model, m->d_model)) {arena_free(&arena); return 0;}
    if (!projection(f, m, start_offset, K_x, x, K_w, K_b, m->d_model, kv_width)) {arena_free(&arena); return 0;}
    if (!projection(f, m, start_offset, V_x, x, V_w, V_b, m->d_model, kv_width)) {arena_free(&arena); return 0;}

    apply_rope(Q_x, m->seq_len, m->head_count, head_dim, rt);
    apply_rope(K_x, m->seq_len, m->head_count_kv, head_dim, rt);

    float inv_sqrt_head_dim = 1.f / sqrtf(head_dim);
    for(size_t i = 0; i < m->seq_len * m->d_model ; i++) Q_x[i] *= inv_sqrt_head_dim;

    float *context = Q_x;

    size_t buf_AB_size = m->seq_len * head_dim;
    size_t buf_S_size  = m->seq_len * m->seq_len;
    float *tmp = arena_alloc(&arena, (buf_AB_size * 2 + buf_S_size) * sizeof(float));
    if (!tmp) {arena_free(&arena); return 0;}
    float *head_buf1 = tmp; 
    float *head_buf2 = tmp + buf_AB_size;
    float *score_buf = head_buf2 + buf_AB_size;  

    for (uint32_t q_head = 0; q_head < m->head_count; q_head++) {
        uint32_t kv_head = q_head / group_size;
        
        pack_head(head_buf1, K_x, m->seq_len, kv_width, head_dim, kv_head);
        transp(head_buf2, head_buf1, m->seq_len, head_dim);

        pack_head(head_buf1, Q_x, m->seq_len, m->d_model, head_dim, q_head);
        int err = mat_mult(score_buf, head_buf1, head_buf2, NULL, m->seq_len, head_dim, head_dim, m->seq_len);
        if (!err) {arena_free(&arena); return 0;}

        softmax_causal(score_buf, m->seq_len);

        pack_head(head_buf1, V_x, m->seq_len, kv_width, head_dim, kv_head);
        err = mat_mult(head_buf2, score_buf, head_buf1, NULL, m->seq_len,  m->seq_len, m->seq_len, head_dim);
        if (!err) {arena_free(&arena); return 0;}

        for (size_t t = 0; t < m->seq_len; t++) {
            memcpy(context + t * m->d_model + q_head * head_dim,
                head_buf2 + t * head_dim,
                head_dim * sizeof(float));
        }

    }

    float *W_o = load_matrix(Wo, f, start_offset);
    //float *b_o = load_matrix(Wo_b, f, start_offset);
    float *b_o = NULL;
    if (!W_o){ free(W_o); free(b_o); arena_free(&arena); return 0;}

    int ok = mat_mult(out, context, W_o, b_o, m->seq_len, m->d_model, m->d_model, m->d_model);
    free(W_o); free(b_o); arena_free(&arena);

    return ok;

}

float *feed_forward(FILE *f, Model *m,long start_offset,
            gguf_tensor_info_t *W_up,
            gguf_tensor_info_t *W_gate,
            gguf_tensor_info_t *W_down,
            float *x)
{
    float *W2 = load_matrix(W_gate, f, start_offset);
    if (!W2) return NULL;
    if (W_gate->n_dimensions != 2 || W_gate->dimensions[0] != m->d_model){
        fprintf(stderr, "ffn_gate shape should be d_model * feed_forward_length\n");
        free(W2);
        return NULL;
    }
    const uint32_t dim_gate = W_gate->dimensions[1];

    float *gate = malloc(m->seq_len * dim_gate * sizeof(float));
    if (!gate) { free(W2); return NULL; }

    int err = mat_mult(gate, x, W2, NULL, m->seq_len, m->d_model, m->d_model, dim_gate);
    free(W2);
    if (!err) { free(gate); return NULL; }

    float *W1 = load_matrix(W_up, f, start_offset);
    if (!W1) { free(gate); return NULL; }
    if (W_up->n_dimensions != 2 || W_up->dimensions[0] != m->d_model) {
        fprintf(stderr, "ffn_up shape should be d_model * feed_forward_length\n");
        free(gate); free(W1);
        return NULL;
    }
    const uint32_t dim_up = W_up->dimensions[1];
    if (dim_gate != dim_up) {
        fprintf(stderr, "ffn_up and ffn_gate should have the same feed_forward_length\n");
        free(gate); free(W1);
        return NULL;
    }

    float *up = malloc(m->seq_len * dim_up * sizeof(float));
    if (!up) { free(gate); free(W1); return NULL; }

    err = mat_mult(up, x, W1, NULL, m->seq_len, m->d_model, m->d_model, dim_up);
    free(W1);
    if (!err) { free(gate); free(up); return NULL; }

    for (size_t i = 0; i < m->seq_len; i++)
        for (size_t j = 0; j < dim_gate; j++)
            gate[i * dim_gate + j] *= up[i * dim_gate + j] / (1.0f + expf(-gate[i * dim_gate + j]));

    free(up);

    float *W3 = load_matrix(W_down, f, start_offset);
    if (!W3) { free(gate); return NULL; }
    if (W_down->n_dimensions != 2 || W_down->dimensions[0] != dim_up || W_down->dimensions[1] != m->d_model){
        fprintf(stderr, "ffn_down shape should be d_model * feed_forward_length\n");
        free(gate); free(W3);
        return NULL;
    }
    uint32_t dim_down = W_down->dimensions[0];

    float *out = malloc(m->seq_len * m->d_model * sizeof(float));
    if (!out) { free(gate); free(W3); return NULL; }

    err = mat_mult(out, gate, W3, NULL, m->seq_len, dim_gate, dim_down, m->d_model);
    free(W3); free(gate);
    if (!err) { free(out); return NULL; }

    return out;
}

gguf_tensor_info_t * get_tensor(gguf_header_t *header, const char *fmt, size_t idx) 
{
    char buff[64]; 
    snprintf(buff, sizeof(buff), fmt, idx);
    gguf_tensor_info_t *target = tensor_lookup(header, buff); 
    if (!target) { 
        fprintf(stderr, "Error: Tensor %s not found\n", buff); 
        return NULL; 
    } 
    return target;
}

float *transformer(Arena *arena, FILE *f, gguf_header_t *header, uint32_t *tokens, Model *m, RopeTable rt, size_t vocab_size)
{
    gguf_metadata_kv_t *head_count_kv_m = metadata_lookup(header, "qwen2.attention.head_count_kv");
    gguf_metadata_kv_t *eps = metadata_lookup(header, "qwen2.attention.layer_norm_rms_epsilon");
    gguf_metadata_kv_t *block_count = metadata_lookup(header, "qwen2.block_count");
    if (!head_count_kv_m || head_count_kv_m->value_type != GGUF_METADATA_VALUE_TYPE_UINT32  ||
        !eps || eps->value_type != GGUF_METADATA_VALUE_TYPE_FLOAT32 ||
        !block_count || block_count->value_type != GGUF_METADATA_VALUE_TYPE_UINT32) return 0;

    m->head_count_kv = head_count_kv_m->value.uint32;
    float epsilon = eps->value.float32;

    gguf_tensor_info_t *embed_tensor = tensor_lookup(header,"token_embd.weight");
    if (!embed_tensor) {
        fprintf(stderr, "Error: Tensor token_embd.weight not found\n"); 
        return NULL;
    }
    float *x = embedding_forward(f, embed_tensor, tokens, m, header->tensor_data_offset, arena);

    size_t size = m->seq_len * m->d_model;
    float *tmp = arena_alloc(arena, 2 * size * sizeof(float));
    float *residual = tmp + size;
    if (!tmp) return NULL;
    

    gguf_tensor_info_t *K_w, *V_w, *Q_w, *K_b, *V_b, *Q_b, *Wo, *Wo_b, *norm_weight, *norm_weight_ffn, *W_up, *W_gate, *W_down;
    for(size_t i = 0; i < block_count->value.uint32; i++){
        if (!(norm_weight = get_tensor(header, "blk.%zu.attn_norm.weight", i))) {free(tmp); return NULL;};
        float *g = load_matrix(norm_weight, f, header->tensor_data_offset);
        memcpy(residual, x, size);
        RMSNorm(m, x, g, epsilon);
        free(g);

        if (!(K_w = get_tensor(header, "blk.%zu.attn_k.weight", i))) return NULL;
        if (!(V_w = get_tensor(header, "blk.%zu.attn_v.weight", i))) return NULL;
        if (!(Q_w = get_tensor(header, "blk.%zu.attn_q.weight", i))) return NULL;

        if (!(K_b = get_tensor(header, "blk.%zu.attn_k.bias", i))) return NULL;
        if (!(V_b = get_tensor(header, "blk.%zu.attn_v.bias", i))) return NULL;
        if (!(Q_b = get_tensor(header, "blk.%zu.attn_q.bias", i))) return NULL;

        if (!(Wo = get_tensor(header, "blk.%zu.attn_output.weight", i))) return NULL;
        //if (!get_tensor(Wo_b, header, "blk.%zu.attn_output.bias", i))  return NULL;
        Wo_b = NULL;

        int err = attention_forward(f, m, header->tensor_data_offset, K_w, V_w, Q_w,
            K_b, V_b, Q_b, Wo, Wo_b,
            x, tmp, rt);
        
        if (!err) return NULL;

        for(size_t j = 0; j < m->seq_len * m->d_model; j++)
            x[j] = residual[j] + tmp[j];
        memcpy(residual, x, size);

        if (!(norm_weight_ffn = get_tensor(header, "blk.%zu.ffn_norm.weight", i))) return NULL;
        float *g2 = load_matrix(norm_weight_ffn, f, header->tensor_data_offset);
        if (!g2) return NULL;
        RMSNorm(m, x, g2, epsilon);
        free(g2);

        if (!(W_up = get_tensor(header, "blk.%zu.ffn_up.weight", i))) return NULL;
        if (!(W_gate = get_tensor(header, "blk.%zu.ffn_gate.weight", i))) return NULL;
        if (!(W_down = get_tensor(header, "blk.%zu.ffn_down.weight", i))) return NULL;

        float *ffn = feed_forward(f, m, header->tensor_data_offset, W_up, W_gate, W_down, x);
        if (!ffn) return NULL;

        for(size_t j = 0; j < m->seq_len * m->d_model; j++)
            x[j] = residual[j] + ffn[j];

        free(ffn);

    }
    arena_pop(arena, tmp, size);

    gguf_tensor_info_t *output_norm = tensor_lookup(header, "output_norm.weight");
    if (!output_norm) { fprintf(stderr, "Error: output_norm.weight not found\n"); return NULL; }
    float *g_out = load_matrix(output_norm, f, header->tensor_data_offset);
    if (!g_out) return NULL;
    RMSNorm(m, x, g_out, epsilon);
    free(g_out);

    float *last = x + (m->seq_len - 1) * m->d_model;
    float *w_out = load_matrix(embed_tensor, f, header->tensor_data_offset);
    if (!w_out) return NULL;

    float *out = malloc(vocab_size * sizeof(float));
    for (size_t j = 0; j < vocab_size; j++) {
        float sum = 0.f;
        for (size_t k = 0; k < m->d_model; k++) {
            sum += last[k] * w_out[k * vocab_size + j];
        }
        out[j] = sum;
    }
    free(w_out);

    return out;

}