#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdint.h>
#include "hashmap.h"
#include "allocator.h"
#include "header.h"

#define TOKENIZER_MAX_SPECIALS 256
#define TOKENIZER_MAX_UTF8_BYTES 8

typedef struct {
    size_t token_count;
    size_t merge_count;

    StrMap token_to_id;
    char **id_to_token;
    StrMap merge_rank;

    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t pad_id;
    uint32_t unk_id;

    const char *special_tokens[TOKENIZER_MAX_SPECIALS];
    uint32_t special_ids[TOKENIZER_MAX_SPECIALS];
    size_t special_count;
} Tokenizer;

int tokenizer_init(Tokenizer *t, Arena *arena, const gguf_header_t *header, const char *filename);
int tokenizer_encode(Tokenizer *t,
                     const char *text,
                     uint32_t **out_tokens,
                     size_t *out_count,
                     int add_bos);
void tokenizer_dump(const Tokenizer *t);

#endif
