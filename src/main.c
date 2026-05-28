#include "../include/baseline.h"
#include <stdlib.h>

int main(int argc, char **argv)
{
    char *file_in = NULL;
    if (argc != 2) return 0;
    file_in = argv[1];

    int err = encode(file_in);
    if(!err) return 1;
    return 0;
}
