#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include <stdint.h>
#include <stdio.h>

typedef struct ArenaBlock {
    struct ArenaBlock *next;
    size_t             used;
    size_t             cap;
    uint8_t           data[];
} ArenaBlock;

typedef struct {
    ArenaBlock *head;
} Arena;


void *arena_alloc(Arena *a, size_t size);
void arena_free(Arena *a);
void arena_print_usage(Arena *a);

#endif
