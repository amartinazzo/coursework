/**

mini desafio 2 - hyper threading

**/


#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define n_calc 8
#define n_cores 2
#define n_matrix 2000

pthread_t *threads;
pthread_mutex_t lock;

clock_t start, end;
double cpu_time_used;

int **A;
int **B;
int **C;

void init_matrices();
void *make_calculations(void *t);
void transpose(int **A);

int main(void) {
    long t;
    int error, threads_started;

    init_matrices();

    threads = malloc(sizeof(pthread_t)*n_calc);

    // modo hyperthreading

    printf("multiplicação de matriz em hyperthreading\n");

    start = clock();

    for(t=0; t < n_calc; t++) {
        error = pthread_create(&threads[t], NULL, make_calculations, (void *) t);
        if(error) {
            printf("pthread_create() returned %d\n", error);
            exit(1);
        }
    }

    for (t=0; t < n_calc; t++) {
        pthread_join(threads[t], NULL);  
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("tempo gasto: %0.2fs\n\n", cpu_time_used);


    // modo simples (threads duas a duas)

    printf("multiplicação de matriz em duas threads por vez\n");

    start = clock();

    threads_started = 0;
    while(threads_started < n_calc) {
        for(t=0; t < n_cores; t++) {
            error = pthread_create(&threads[t], NULL, make_calculations, (void *) t);
            if(error) {
                printf("pthread_create() returned %d\n", error);
                exit(1);
            }
        }

        for (t=0; t < n_cores; t++) {
            pthread_join(threads[t], NULL);  
        }
        threads_started += n_cores;
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("tempo gasto: %0.2fs\n\n", cpu_time_used);


    // modo sequencial (sem threads)

    printf("multiplicação de matriz em uma única thread\n");

    start = clock();
    for(t=0; t < n_calc; t++) {
        make_calculations(0);
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("tempo gasto: %0.2fs\n\n", cpu_time_used);

    return 0;
}

void init_matrices() {
    int i, j;
    A = (int **)malloc(sizeof(int *)*n_matrix);
    B = (int **)malloc(sizeof(int *)*n_matrix);
    C = (int **)malloc(sizeof(int *)*n_matrix);

    for (i=0; i < n_matrix; i++) {
        A[i] = (int *)malloc(sizeof(int)*n_matrix);
        B[i] = (int *)malloc(sizeof(int)*n_matrix);
        C[i] = (int *)malloc(sizeof(int)*n_matrix);
        for (j = 0; j < n_matrix; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }
    }
}

void transpose(int **A) {
    int i, j;
    int **tmp;
    tmp = (int **)malloc(sizeof(int *)*n_matrix);
    for(i=0; i < n_matrix; i++) tmp[i] = (int *)malloc(sizeof(int)*n_matrix);

    for(i=0; i < n_matrix; ++i) {
        for(j=0; j < n_matrix; ++j) {
            tmp[j][i] = A[i][j];
        }
    }
    for(i=0; i < n_matrix; ++i) {
        for(j=0; j < n_matrix; ++j) {
            A[i][j] = tmp[i][j];
        }
    }
}

void *make_calculations(void *t){
    long thread_id = (long) t;
    //printf("inicializando #%ld\n", thread_id);
    int i, j, k;

    transpose(B);
    for (i=0; i < n_matrix; i++) {
        for (j=0; j < n_matrix; j++) {
            C[i][j] = 0;
            for (k=0; k < n_matrix; k++) {
                C[i][j] += A[i][k]*B[j][k];
            }
        }
    }
}