#include "../include/allocator.h"
#include "../include/header.h"
#include "../include/tokenizer.h"

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
    if (!tokenizer_init(&t, &arena,  header, file_name)){
         arena_free(&arena);
         return 0;
    }
    tokenizer_dump(&t);

    arena_print_usage(&arena);
    arena_free(&arena);
    return 1;
}
