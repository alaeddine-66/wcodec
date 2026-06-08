#ifndef HASHMAP_H
#define HASHMAP_H

#include <stdio.h>
#include "allocator.h"

typedef struct {
    char *key;
    uint32_t value;
} HashMapEntry;

typedef struct {
    HashMapEntry *entries;
    size_t cap;
    size_t len;
} HashMap_t;

int strmap_init(HashMap_t *m, Arena *arena, size_t initial_cap);
int strmap_put(HashMap_t *m, const char *key_ref, uint32_t value);
int strmap_get(const HashMap_t *m, const char *key, uint32_t *out_value);

#endif
