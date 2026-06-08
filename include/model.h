#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

typedef struct {
    uint32_t seq_len, d_model;
}Model;

#endif