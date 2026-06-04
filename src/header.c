#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "../include/allocator.h"
#include "../include/header.h"
#include <stdbool.h>

int read_string(Arena *arena, gguf_string_t *str, FILE *f)
{
    uint64_t len;
    if (fread(&len, sizeof(len), 1, f) != 1) return 0;

    char *key = arena_alloc(arena, len + 1);
    if (!key) return 0;
    if (fread(key, 1, len, f) != len) return 0;;
    key[len] = '\0';

    str->len = len;
    str->string = key; 
    return 1;
}

int store_ref(DeferredStringArray *ref, FILE *f, uint64_t len)
{

    ref->file_offset = ftell(f);
    ref->count       = len;

    for (uint64_t i = 0; i < len; i++) {
        uint64_t slen;
        if (fread(&slen, sizeof(slen), 1, f) != 1) return 0;
        if (fseek(f, (long)slen, SEEK_CUR) != 0) return 0;
    }
    return 1;

}

int read_token_types_rle(Arena *arena, TokenTypeRLE *out,
                                 FILE *f, uint64_t len) {
    long start = ftell(f);
    uint32_t run_count = 0;
    int32_t prev;
    if (fread(&prev, sizeof(prev), 1, f) != 1) return 0;
    run_count = 1;

    for (uint64_t i = 1; i < len; i++) {
        int32_t v;
        if (fread(&v, sizeof(v), 1, f) != 1) return 0;
        if (v != prev) { run_count++; prev = v; }
    }

    fseek(f, start, SEEK_SET);
    TokenTypeRun *runs = arena_alloc(arena, run_count * sizeof(TokenTypeRun));
    if (!runs) return 0;

    uint32_t ri = 0;
    int32_t cur;
    if (fread(&cur, sizeof(cur), 1, f) != 1) return 0;
    runs[ri].value = cur;
    runs[ri].count = 1;

    for (uint64_t i = 1; i < len; i++) {
        int32_t v;
        if (fread(&v, sizeof(v), 1, f) != 1) return 0;
        if (v == runs[ri].value) {
            runs[ri].count++;
        } else {
            ri++;
            runs[ri].value = v;
            runs[ri].count = 1;
        }
    }

    out->runs      = runs;
    out->run_count = run_count;
    out->total     = len;
    return 1;
}

int read_value(Arena *arena, uint32_t value_type, gguf_metadata_value_t *value, FILE *f, 
            char *key, DeferredStringArray *tokens_ref,
            DeferredStringArray *merges_ref,
            TokenTypeRLE *token_types)
{
    switch (value_type){
        case GGUF_METADATA_VALUE_TYPE_UINT8:{
            uint8_t v;
            if (fread(&v, sizeof(v), 1, f) != 1) return 0;
            value->uint8 = v;
            break;
        }
        case GGUF_METADATA_VALUE_TYPE_INT8:{
            int8_t v;
            if (fread(&v, sizeof(v), 1, f) != 1) return 0;
            value->int8 = v;
            break;
        }    
        case GGUF_METADATA_VALUE_TYPE_UINT16:{
            uint16_t v;
            if (fread(&v, sizeof(v), 1, f) != 1) return 0;
            value->uint16 = v;
            break;
        }
        case GGUF_METADATA_VALUE_TYPE_INT16:{
            int16_t v;
            if (fread(&v, sizeof(v), 1, f) != 1) return 0;
            value->int16 = v;
            break;
        }    
        case GGUF_METADATA_VALUE_TYPE_UINT32:{
            uint32_t v;
            if (fread(&v, sizeof(v), 1, f) != 1) return 0;
            value->uint32 = v;
            break;
        }
        case GGUF_METADATA_VALUE_TYPE_INT32:{
            int32_t v;
            if (fread(&v, sizeof(v), 1, f) != 1) return 0;
            value->int32 = v;
            break;
        }    
        case GGUF_METADATA_VALUE_TYPE_FLOAT32:{
            float v;
            if (fread(&v, sizeof(v), 1, f) != 1) return 0;
            value->float32 = v;
            break;
        }
        case GGUF_METADATA_VALUE_TYPE_BOOL:{
            bool v;
            if (fread(&v, sizeof(v), 1, f) != 1) return 0;
            value->bool_ = v;
            break;
        }
        case GGUF_METADATA_VALUE_TYPE_STRING:{
            return read_string(arena, &value->string, f);
        }
        case GGUF_METADATA_VALUE_TYPE_UINT64:{
            uint64_t v; 
            if (fread(&v, sizeof(v), 1, f) != 1) return 0;
            value->uint64 = v;
            break;
        }
        case GGUF_METADATA_VALUE_TYPE_INT64:{
            int64_t v;
            if (fread(&v, sizeof(v), 1, f) != 1) return 0;
            value->int64 = v;
            break;
        }
        case GGUF_METADATA_VALUE_TYPE_FLOAT64:{
            double v;
            if (fread(&v, sizeof(v), 1, f) != 1) return 0;
            value->float64 = v;
            break;
        }
        case GGUF_METADATA_VALUE_TYPE_ARRAY:{
            uint32_t type;
            if (fread(&type, sizeof(type), 1, f) != 1) return 0;
            uint64_t len;
            if (fread(&len, sizeof(len), 1, f) != 1) return 0;

            int err = 1;
            if (strcmp(key, "tokenizer.ggml.tokens") == 0){
                err = store_ref(tokens_ref, f, len);
            }else if (strcmp(key, "tokenizer.ggml.merges") == 0){
                err = store_ref(merges_ref, f, len);
            }else if (strcmp(key, "tokenizer.ggml.token_type") == 0){
                err = read_token_types_rle(arena, token_types, f, len);
            }else {
                value->array.type  = type;
                value->array.len   = len;
                value->array.array = arena_alloc(arena, len * sizeof(gguf_metadata_value_t));
                if (!value->array.array) {
                    fprintf(stderr, "Error: Allocation failed");
                    return 0;
                }
                for (uint64_t i = 0; i < len; i++) {
                    if (!read_value(arena, type, &value->array.array[i], f, key, tokens_ref, merges_ref, token_types)) {
                        return 0;
                    }
                }
            }
            if(!err) return 0;
            break;
        }
        default:{
            fprintf(stderr, "Error: unknown value type %u\n", value_type);
            return 0;
        }
    }
    return 1;
}


int read_kv(Arena *arena, gguf_metadata_kv_t *metadata_kv, uint64_t kv_count, FILE *f,
    DeferredStringArray *tokens_ref,
    DeferredStringArray *merges_ref,
    TokenTypeRLE *token_types)
{
    for(uint64_t i = 0; i < kv_count; i++){
        
        int err = read_string(arena, &metadata_kv[i].key, f);
        if (!err) return 0;

        uint32_t value_type;
        if (fread(&value_type, sizeof(value_type), 1, f) != 1) return 0;
        metadata_kv[i].value_type = value_type;

        err = read_value(arena, value_type, &metadata_kv[i].value, f, metadata_kv[i].key.string,
                tokens_ref, merges_ref, token_types);
        if (!err) return 0;
    }

    return 1;
}

int read_tensors(Arena *arena, gguf_tensor_info_t *tensors, uint64_t tensor_count, FILE *f)
{
    for(uint64_t i = 0; i < tensor_count; i++){

        int err = read_string(arena, &tensors[i].name, f);
        if (!err) return 0; 

        uint32_t n_dimensions;
        if (fread(&n_dimensions, sizeof(n_dimensions), 1, f) != 1) return 0;
        tensors[i].n_dimensions = n_dimensions;

        tensors[i].dimensions = arena_alloc(arena, n_dimensions * sizeof(uint64_t));
        if (!tensors[i].dimensions) return 0;

        if (fread(tensors[i].dimensions, sizeof(uint64_t), n_dimensions, f) != n_dimensions) return 0;
        if (fread(&tensors[i].type, sizeof(uint32_t), 1, f) != 1) return 0;
        if (fread(&tensors[i].offset, sizeof(uint64_t), 1, f) != 1) return 0;

    }

    return 1;
}

static void find_alignment(uint32_t *alignment, uint64_t kv_count, gguf_metadata_kv_t *metadata_kv)
{
    for(uint64_t i = 0; i < kv_count; i++){
        if((strcmp(metadata_kv[i].key.string, "general.alignment") == 0)
	 && (metadata_kv[i].value_type == GGUF_METADATA_VALUE_TYPE_UINT32)){
	        *alignment = metadata_kv[i].value.uint32;
	        return;
	    }
    }
}

static inline long align_offset(long offset, uint32_t alignment) {
    return offset + (alignment - (offset % alignment)) % alignment;
}

gguf_header_t *read_header(Arena *arena, const char *file_name){
    FILE *f = fopen(file_name, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open file : %s\n", file_name);
        return NULL;
    }

    char magic_num[5];
    size_t n = fread(magic_num, 1, 4, f);
    magic_num[4] = '\0';
    if (n != 4){
        fprintf(stderr, "Error: cannot read magic number");
        fclose(f);
        return NULL;
    }
    if (strcmp(magic_num, "GGUF") != 0){
        fprintf(stderr, "Error: invalid magic number");
        fclose(f);
        return NULL;
    }

    uint32_t version;
    n = fread(&version, sizeof(version), 1, f);
    if ((n != 1) || (version != 3)){
        fprintf(stderr, "Error, incorrect version\n");
        fclose(f);
        return NULL;
    }

    uint64_t tensor_count;
    if (fread(&tensor_count, sizeof(tensor_count), 1, f) != 1){
        fclose(f);
	return NULL;
    }

    uint64_t kv_count;
    if (fread(&kv_count, sizeof(kv_count), 1, f) != 1){
        fclose(f);
	return NULL;
    }

    gguf_header_t *gguf_reader = arena_alloc(arena, sizeof(gguf_header_t));
    if (!gguf_reader){
        fprintf(stderr, "Error: Allocation failed in header.c\n");
        fclose(f);
        return NULL;
    }
    gguf_reader->tensor_count = tensor_count;
    gguf_reader->metadata_kv_count = kv_count;
    gguf_reader->tensors = NULL;

    gguf_reader->metadata_kv = arena_alloc(arena, kv_count * sizeof(gguf_metadata_kv_t));
    if (!gguf_reader->metadata_kv){
        fprintf(stderr, "Error: Allocation failed in header.c\n");
        fclose(f);
        return NULL;
    }

    int err = read_kv(arena, gguf_reader->metadata_kv, kv_count, f,
                    &gguf_reader->tokens_ref, &gguf_reader->merges_ref, &gguf_reader->token_types);
    if (!err){
        fclose(f);
        return NULL;
    }

    gguf_reader->tensors = arena_alloc(arena, tensor_count * sizeof(gguf_tensor_info_t));
    if (!gguf_reader->tensors){
        fprintf(stderr, "Error: Allocation failed in header.c \n");
        fclose(f);
        return NULL;
    }

    err = read_tensors(arena, gguf_reader->tensors, tensor_count, f);
    if (!err){
        fclose(f);
        return NULL;
    }

    uint32_t alignment = 32;
    find_alignment(&alignment, kv_count, gguf_reader->metadata_kv);
    long offset = ftell(f);
    gguf_reader->tensor_data_offset = align_offset(offset, alignment);

    fclose(f);
    return gguf_reader;

}

