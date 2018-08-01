#include <stdio.h>
#include <stdlib.h>
#define maxbrightness 255

int main(int argc, char **argv) {
	if(argc < 3) {
		printf("são necessários dois parâmetros: as dimensões da matriz MxN");
		exit(0);
	}

	int i, j;
	int M = atoi(argv[1]);
	int N = atoi(argv[2]);

	// int **I = malloc((int **) sizeof(int)*M);
	// for(i=0; i < M; i++) I[i] = malloc((int *) sizeof(int)*N);

	printf("P2\n%d %d\n%d\n", M, N, maxbrightness);

	for(i=0; i < M; i++) {
		for(j=0; j < N; j++) {
			// I[i][j] = rand() % maxbrightness;
			printf("%d ", rand() % maxbrightness);
		}
		printf("\n");
	}

}