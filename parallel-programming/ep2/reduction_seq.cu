#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "functions.h"
#include "reduction_seq.h"


int* reduction_seq(const char filename[], int matrix_order)
{
  int **x;
  int* y;
  int n_els = matrix_order*matrix_order;
  int n_mat = get_n_mat(filename);
  int val1, val2, val3;
  FILE *fp;

  /* allocate memory for y */
  y = (int*)malloc(n_els * sizeof(int));


  /* allocate memory for x */
  x = (int**)malloc(n_els*sizeof(int*));
  for(int i = 0; i<n_els; i++)
  {
    x[i] = (int*)malloc(n_mat * sizeof(int));
  }

  /* read matrix list */
  fp = fopen(filename, "r");
  fscanf(fp, "%*s");
  fscanf(fp, "%*s");
  for(int i=0; i < n_mat; i++)
  {
    for(int j=0; j < matrix_order; j++)
    {
        fscanf(fp, "%d %d %d", &val1, &val2, &val3);
        x[matrix_order*j][i] = val1;
        x[matrix_order*j+1][i] = val2;
        x[matrix_order*j+2][i] = val3;
    }
    fscanf(fp, "%*s");  // skip line
  }
  fclose(fp);

  /* initialize y with first matrix */
  for(int j=0; j < matrix_order; j++)
  {
      y[matrix_order*j] = x[matrix_order*j][0];
      y[matrix_order*j+1] = x[matrix_order*j+1][0];
      y[matrix_order*j+2] = x[matrix_order*j+2][0];
  }

  /* sequential reduction */
  for(int i=1; i < n_mat; i++)
  {
    for(int j=0; j < matrix_order; j++)
    {
        if(x[matrix_order*j][i] < y[matrix_order*j])
          y[matrix_order*j] = x[matrix_order*j][i];
        if(x[matrix_order*j+1][i] < y[matrix_order*j+1])
          y[matrix_order*j+1] = x[matrix_order*j+1][i];
        if(x[matrix_order*j+2][i] < y[matrix_order*j+2])
          y[matrix_order*j+2] = x[matrix_order*j+2][i];
    }
  }

  return y;
}
