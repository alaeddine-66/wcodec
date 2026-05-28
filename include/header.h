#ifndef HEADER_H
#define HEADER_H 

#include "../include/allocator.h"

gguf_header_t *read_header(Arena *arena, char *file_name);

#endif
