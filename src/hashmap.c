#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include "../include/hashmap.h"
#include "../include/allocator.h"

static uint64_t fnv1a_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

int strmap_init(HashMap_t *m, Arena *arena, size_t initial_cap) {
    if (initial_cap < 8) initial_cap = 8;
    size_t cap = 8;
    size_t min = (initial_cap * 10 + 6) / 7 + 1;
    while (cap < min) cap <<= 1;
    //printf("cap=%zu\n", cap);
    m->entries = (HashMapEntry *)arena_alloc(arena, cap * sizeof(HashMapEntry));
    if (!m->entries) return 0;
    memset(m->entries, 0, cap * sizeof(HashMapEntry));
    m->cap = cap;
    m->len = 0;
    return 1;
}

int strmap_put(HashMap_t *m, const char *key_owned, uint32_t value) {
    assert((m->len + 1) * 10 < m->cap * 7);

    uint64_t h = fnv1a_hash(key_owned);
    size_t idx = (size_t)h & (m->cap - 1);
    while (m->entries[idx].key) {
        if (strcmp(m->entries[idx].key, key_owned) == 0) {
            m->entries[idx].value = value;
            return 1;
        }
        idx = (idx + 1) & (m->cap - 1);
    }
    m->entries[idx].key = (char *)key_owned;
    m->entries[idx].value = value;
    m->len++;
    return 1;
}

int strmap_get(const HashMap_t *m, const char *key, uint32_t *out_value) {
    if (!m || !m->entries || m->cap == 0) return 0;
    uint64_t h = fnv1a_hash(key);
    size_t idx = (size_t)h & (m->cap - 1);
    size_t start = idx;
    while (m->entries[idx].key) {
        if (strcmp(m->entries[idx].key, key) == 0) {
            if (out_value) *out_value = m->entries[idx].value;
            return 1;
        }
        idx = (idx + 1) & (m->cap - 1);
        if (idx == start) break;
    }
    return 0;
}
