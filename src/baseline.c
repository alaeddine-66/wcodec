#include "../include/allocator.h"
#include "../include/header.h"
#include "../include/tokenizer.h"
#include "../include/hashmap.h"
#include <stdlib.h>

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
    //tokenizer_dump(&t);

    uint32_t *tokens = NULL;
    size_t n = 0;

    if (!tokenizer_encode(&t, "hello world", &tokens, &n, 1)) {
        fprintf(stderr, "encode failed\n");
        return 0;
    }



    for (size_t i = 0; i < n; i++) {
        printf("%u ", tokens[i]);
    }
    printf("\n");
    free(tokens);

    arena_print_usage(&arena);
    arena_free(&arena);
    
    printf("EXIT SUCCESS\n");
    printf("%zu\n", sizeof(size_t));
    return 1;
}
