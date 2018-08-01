/**

mini EP1 - evidenciando o uso do cachê

**/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define N 1024
#define maxvalue 1000

clock_t start, end;
double cpu_time_used;

void calc_performance(void multiply_func(int **, int **, int **), int **A, int **B, int **C);
void multiply(int **A, int **B, int **C);
void multiply_transpose(int **A, int **B, int **C);
void transpose(int **A);

int main(int argc, char **argv) {
    int i, j;
    int mode = -1;

    int **A;
    int **B;
    int **C;

    if(argv[1] != NULL) mode = atoi(argv[1]);

    // aloca matrizes e as inicializa com valores aleatórios

    A = (int **)malloc(sizeof(int *)*N);
    B = (int **)malloc(sizeof(int *)*N);
    C = (int **)malloc(sizeof(int *)*N);

    for (i=0; i < N; i++) {
        A[i] = (int *)malloc(sizeof(int)*N);
        B[i] = (int *)malloc(sizeof(int)*N);
        C[i] = (int *)malloc(sizeof(int)*N);
        for (j = 0; j < N; j++) {
            A[i][j] = rand() % maxvalue;
            B[i][j] = rand() % maxvalue;
        }
    }

    // roda cada um dos métodos e calcula o tempo

    printf("N = %d\n", N);

    if(mode == -1) {
        printf("multiplicação simples:\n");
        calc_performance(multiply, A, B, C);
        printf("multiplicação com B':\n");
        calc_performance(multiply_transpose, A, B, C);
    } else if(mode == 0) {
        printf("multiplicação simples:\n");
        calc_performance(multiply, A, B, C);
    } else if (mode == 1) {
        printf("multiplicação com B':\n");
        calc_performance(multiply_transpose, A, B, C);
    }

    return 0;
}

void transpose(int **A) {
    int i, j;
    int **tmp;

    tmp = (int **)malloc(sizeof(int *)*N);

    for(i=0; i < N; i++) tmp[i] = (int *)malloc(sizeof(int)*N);

    for(i=0; i<N; ++i) {
        for(j=0; j<N; ++j) {
            tmp[j][i] = A[i][j];
        }
    }

    for(i=0; i<N; ++i) {
        for(j=0; j<N; ++j) {
            A[i][j] = tmp[i][j];
        }
    }
}

void multiply(int **A, int **B, int **C) {
    int i, j, k;
    for (i=0; i < N; i++) {
        for (j=0; j < N; j++) {
            C[i][j] = 0;
            for (k=0; k < N; k++) {
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

void multiply_transpose(int **A, int **B, int **C) {
    int i, j, k;
    transpose(B);
    for (i=0; i < N; i++) {
        for (j=0; j < N; j++) {
            C[i][j] = 0;
            for (k=0; k < N; k++) {
                C[i][j] += A[i][k]*B[j][k];
            }
        }
    }
}

void calc_performance(void multiply_func(int **, int **, int **), int **A, int **B, int **C) {
    start = clock();
    multiply_func(A, B, C);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("tempo: %.1e\n", cpu_time_used);
}
