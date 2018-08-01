/**
 
mini desafio 1 - evidenciando branch predictions
 
**/
 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define N 2048
#define threshold N/2
 
clock_t start, end;
double cpu_time_used;

void calc_performance(int vector[N], int sort);
int compare_int(const void *a, const void *b);
int sum(int vector[N]);
 
int main(void) {
    int i;
    int V[N];

    printf("N = %d, threshold = %d\n", N, threshold);

    printf("soma em vetor desordenado (branch prediction menos eficiente):\n");
    calc_performance(V, 0);
    printf("soma em vetor ordenado (branch prediction mais eficiente):\n");
    calc_performance(V, 1);

    printf("\n");
 
    return 0;
}

int compare_int(const void *a, const void *b) {
    int *x = (int *) a;
    int *y = (int *) b;
    return *x - *y;
}
 
int sum(int vector[N]) {
    int sum = 0;
    for(int i=0; i < N; i++) {
        if(vector[i] > threshold) sum++;
    }
    return sum;
}
 
void calc_performance(int vector[N], int sort) {
    if(sort == 1) qsort(vector, N, sizeof(vector[0]), compare_int);
    start = clock();
    sum(vector);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("tempo: %f\n", cpu_time_used);
}