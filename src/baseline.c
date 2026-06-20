#include "../include/allocator.h"
#include "../include/header.h"
#include "../include/tokenizer.h"
#include "../include/hashmap.h"
#include "../include/model.h"
#include "../include/rope.h"
#include <stdlib.h>

#define MAX_NEW_TOKENS 250

static size_t argmax(const float *arr, size_t n)
{
    size_t best = 0;
    float best_val = arr[0];
    for (size_t i = 1; i < n; i++) {
        if (arr[i] > best_val) {
            best_val = arr[i];
            best = i;
        }
    }
    return best;
}

int encode(char *file_name)
{
    Arena arena = {0};
    gguf_header_t *header = read_header(&arena, file_name);
    if (!header) {
        arena_free(&arena);
        return 0;
    }
    printf("tensor_count = %llu\n", header->tensor_count);
    printf("metadata_kv_count = %llu\n", header->metadata_kv_count);

    Tokenizer t = {0};
    if (!tokenizer_init(&t, &arena, header, file_name)) {
        arena_free(&arena);
        return 0;
    }

    uint32_t *tokens = NULL;
    size_t n = 0;

    if (!tokenizer_encode(&t,
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "who are you ?<|im_end|>\n"
        "<|im_start|>assistant\n",
        &tokens, &n, 0)){
        fprintf(stderr, "encode failed\n");
        arena_free(&arena);
        return 0;
    }

    /* [d_model, vocab_size] */
    gguf_tensor_info_t *embed_tensor = tensor_lookup(header, "token_embd.weight");
    if (!embed_tensor || embed_tensor->n_dimensions != 2) {
        fprintf(stderr, "token_embd.weight not found or invalid\n");
        free(tokens);
        arena_free(&arena);
        return 0;
    }
    size_t vocab_size = embed_tensor->dimensions[1];

    FILE *f = fopen(file_name, "rb");
    if (!f) {
        free(tokens);
        arena_free(&arena);
        return 0;
    }

    size_t cap = n + MAX_NEW_TOKENS;
    uint32_t *tmp = realloc(tokens, cap * sizeof(uint32_t));
    if (!tmp) {
        fprintf(stderr, "realloc failed\n");
        fclose(f);
        free(tokens);
        arena_free(&arena);
        return 0;
    }
    tokens = tmp;

    printf("input tokens: ");
    for (size_t i = 0; i < n; i++) {
        printf("%u ", tokens[i]);
    }
    printf("\n\n");

    gguf_metadata_kv_t *rope_theta_m = metadata_lookup(header, "qwen2.rope.freq_base");
    gguf_metadata_kv_t *embedding_length = metadata_lookup(header, "qwen2.embedding_length");
    gguf_metadata_kv_t *head_count_m = metadata_lookup(header, "qwen2.attention.head_count");
    gguf_metadata_kv_t *head_count_kv_m = metadata_lookup(header, "qwen2.attention.head_count_kv");
    gguf_metadata_kv_t *block_count = metadata_lookup(header, "qwen2.block_count");
    if (
        !head_count_m || head_count_m->value_type != GGUF_METADATA_VALUE_TYPE_UINT32 ||
        !rope_theta_m || rope_theta_m->value_type != GGUF_METADATA_VALUE_TYPE_FLOAT32 ||
        !embedding_length || embedding_length->value_type != GGUF_METADATA_VALUE_TYPE_UINT32 ||
        !head_count_kv_m || head_count_kv_m->value_type != GGUF_METADATA_VALUE_TYPE_UINT32 ||
        !block_count || block_count->value_type != GGUF_METADATA_VALUE_TYPE_UINT32
    ){  fprintf(stderr, "metadata not found\n");
        fclose(f);
        free(tokens);
        arena_free(&arena);
        return 0;
    }

    Model m = {0};
    m.d_model = embedding_length->value.uint32;
    m.head_count = head_count_m->value.uint32;
    m.head_count_kv = head_count_kv_m->value.uint32;
    m.block_count = block_count->value.uint32;
    uint32_t head_dim = m.d_model / m.head_count;
    uint32_t kv_width = m.head_count_kv * head_dim;
    float rope_theta= rope_theta_m->value.float32;
    RopeTable rt = build_rope_table(&arena, cap, head_dim, rope_theta);

    m.seq_len = n;
    /* allocation des pointeurs de couches */
    m.K_cache = arena_alloc(&arena, m.block_count * sizeof(float *));
    m.V_cache = arena_alloc(&arena, m.block_count * sizeof(float *));

    if (!m.K_cache || !m.V_cache) {
        fprintf(stderr, "K/V cache pointer allocation failed\n");
        fclose(f); free(tokens); arena_free(&arena);
        return 0;
    }

    /* allocation de chaque couche */
    for (size_t l = 0; l < m.block_count; l++) {

        m.K_cache[l] = arena_alloc(&arena, cap * kv_width * sizeof(float));
        m.V_cache[l] = arena_alloc(&arena, cap * kv_width * sizeof(float));

        if (!m.K_cache[l] || !m.V_cache[l]) {
            fprintf(stderr, "KV cache allocation failed at layer %zu\n", l);
            fclose(f); free(tokens); arena_free(&arena);
            return 0;
        }
    }
    size_t pos = 0;
    for (size_t step = 0; step < MAX_NEW_TOKENS; step++) {

        float *out = transformer(&arena, f, header, &tokens[pos], &m, rt, vocab_size, pos);
        if (!out) {
            fprintf(stderr, "transformer failed at step %zu\n", step);
            break;
        }

        uint32_t next_id = (uint32_t)argmax(out, vocab_size);
        free(out);

        char *decoded = NULL;
        if (tokenizer_decode(&t, &arena, &next_id, 1, &decoded)) {
            printf("%s", decoded);
            free(decoded);
        } else {
            printf("[%zu] id=%u  (decode failed)\n", step, next_id);
        }

        if (next_id == t.eos_id) {
            printf("\n(EOS reached)\n");
            break;
        }
        tokens[n] = next_id;  
        pos = n;              
        n++;                 
        m.seq_len = 1;       
    }

    printf("\nfull sequence: ");
    for (size_t i = 0; i < n; i++) {
        printf("%u ", tokens[i]);
    }
    printf("\n");

    free(tokens);
    fclose(f);

    arena_print_usage(&arena);
    arena_free(&arena);

    printf("EXIT SUCCESS\n");
    return 1;
}
