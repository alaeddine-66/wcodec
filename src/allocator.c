#include "../include/allocator.h"
#include <stdlib.h>

#define ARENA_BLOCK_SIZE (1 * 1024 * 1024)

static ArenaBlock *arena_new_block(size_t min_size) {
    size_t cap = min_size > ARENA_BLOCK_SIZE ? min_size : ARENA_BLOCK_SIZE;
    ArenaBlock *b = malloc(sizeof(ArenaBlock) + cap);
    if (!b) return NULL;
    b->next = NULL;
    b->used = 0;
    b->cap  = cap;
    return b;
}


void *arena_alloc(Arena *a, size_t size) {
    size = (size + 7) & ~(size_t)7;

    if (a->head && a->head->used + size <= a->head->cap) {
        void *ptr = a->head->data + a->head->used;
        a->head->used += size;
        return ptr;
    }

    ArenaBlock *b = arena_new_block(size);
    if (!b) {
        fprintf(stderr, "Error, malloc failed\n");
        return NULL;
    }

    if (size > ARENA_BLOCK_SIZE / 2 && a->head) {
        b->next    = a->head->next;
        a->head->next = b;
    } else {
        b->next = a->head;
        a->head = b;
    }

    void *ptr = b->data;
    b->used   = size;
    return ptr;
}

void arena_free(Arena *a) {
    ArenaBlock *b = a->head;
    while (b) {
        ArenaBlock *next = b->next;
        free(b);
        b = next;
    }
    a->head = NULL;
}

void arena_pop(Arena *a, void *ptr, size_t size) {
    if (!a || !ptr) return;

    size = (size + 7) & ~(size_t)7;

    if (a->head && ptr == (void *)(a->head->data + a->head->used - size)) {
        a->head->used -= size;

        if (a->head->used == 0 && a->head->next) {
            ArenaBlock *old_head = a->head;
            a->head = old_head->next;
            free(old_head);
        }
        return;
    }

    if (a->head && a->head->next && ptr == (void *)a->head->next->data) {
        ArenaBlock *large_block = a->head->next;
        a->head->next = large_block->next;
        free(large_block);
        return;
    }
}

void arena_print_usage(Arena *a) {
    size_t total_used = 0;
    size_t total_cap = 0;
    int nb_blocks = 0;

    ArenaBlock *b = a->head;

    while (b) {
        total_used += b->used;
        printf("block %d usage : %zu\n", nb_blocks, b->used);
        total_cap  += b->cap;
        nb_blocks++;
        b = b->next;
    }

    printf("Arena usage:\n");
    printf("  blocks      : %d\n", nb_blocks);
    printf("  used        : %zu bytes\n", total_used);
    printf("  capacity    : %zu bytes\n", total_cap);
    printf("  fragmentation: %zu bytes\n", total_cap - total_used);
}
