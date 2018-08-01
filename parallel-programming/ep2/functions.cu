#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "functions.h"

void print_matrix(int** matrix, int n_rows, int n_cols)
{
	for(int i=0; i<n_rows; i++)
	{
		for(int j=0; j<n_cols; j++)
		{
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
}

void print_vector(int* vector, int D)
{
	int n_els = D*D;
	for(int i=0; i<n_els; i++)
	{
		printf("%d ", vector[i]);
		if((i+1)%D == 0)
			printf("\n");
	}
	printf("\n");
}

int get_n_mat(const char filename[])
{
  int n_mat;

  FILE *fp;
  fp = fopen(filename, "r");

  if (fp == NULL)
  {
      printf ("Error opening the file\n\n'");
      exit(EXIT_FAILURE);
  }

  fscanf(fp, "%d", &n_mat);
  fclose(fp);
  return n_mat;
}

int assert_vector(int* a, int* b, int size)
{
  for(int i=0; i<size; i++){
    if(a[i] != b[i])
      return 1;
  }
  return 0;
}

void print_test_result(const char description[], int result)
{
    if(result == 1){
      printf("Teste: %s [Falhou]\n", description);
      return;
    }
    printf("Teste: %s [OK]\n", description);
    return;
}

int randMToN(int M, int N)
{
    return M + (rand() / ( RAND_MAX / (N-M) ) ) ;
}

void write_matrix_list(int n_mat, const char filename[], int matrix_order)
{
  /* open file */
  FILE *f = fopen(filename, "w");
  if (f == NULL)
  {
      printf("Error opening file!\n");
      exit(1);
  }
  /* print num matrizes */
  fprintf(f, "%d\n", n_mat);
  fprintf(f, "***\n");
  /* write matrix */
  for(int n=1; n<=n_mat; n++)
  {
    for(int i=1; i<=matrix_order; i++)
    {
        fprintf(f, "%d %d %d\n", randMToN(1,10000000), randMToN(1,10000000), randMToN(1,10000000));
    }
    fprintf(f, "***\n");
  }
  /* close file */
  fclose(f);
}

long time_elapsed (struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
}

void print_performance_test_result(
  const char* description, long time_elapsed_cuda, long time_elapsed_seq
)
{
  printf("Teste: %s\n", description);
  printf("Tempo Cuda: %ld us\n", time_elapsed_cuda);
  printf("Tempo Sequencial: %ld us\n", time_elapsed_seq);
  if(time_elapsed_cuda < time_elapsed_seq)
    printf("A implementação Cuda foi mais rápida em: %ld us\n\n", time_elapsed_seq-time_elapsed_cuda);
  else if (time_elapsed_cuda > time_elapsed_seq)
    printf("A implementação Sequencial foi mais rápida em: %ld us\n\n", time_elapsed_cuda-time_elapsed_seq);
  else
    printf("As duas implementações tiveram o mesmo tempo de execução\n\n");
}
