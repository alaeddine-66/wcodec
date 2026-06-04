#ifndef HASHMAP_H
#define HASHMAP_H

#include <stdio.h>
#include "allocator.h"

typedef struct {
    char *key;
    uint32_t value;
    unsigned char used;
} StrMapEntry;

typedef struct {
    StrMapEntry *entries;
    size_t cap;
    size_t len;
} StrMap;

int strmap_init(StrMap *m, Arena *arena, size_t initial_cap);
int strmap_put(StrMap *m, const char *key_ref, int value);
int strmap_get(const StrMap *m, const char *key, int *out_value);

#endif
