/**
EP2 - Redução em CUDA
**/

#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "functions.h"
#include "reduction_cuda.h"
#include "reduction_seq.h"

#define MATRIX_ORDER 3 // ordem das matrizes (quadradas)

int main(int argc, char *argv[])
{
    const char* filename = argv[1];

    int *y_cuda = reduction_cuda(filename, MATRIX_ORDER);
    int *y_seq = reduction_seq(filename, MATRIX_ORDER);

    // printf("Cuda reduction result:\n");
    print_vector(y_cuda, MATRIX_ORDER);
    // printf("Sequential reduction result:\n");
    // print_vector(y_seq, MATRIX_ORDER);

    return 0;
}
