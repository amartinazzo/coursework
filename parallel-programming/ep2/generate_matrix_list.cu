#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "functions.h"

#define MATRIX_ORDER 3 // ordem das matrizes (quadradas)

int main(int argc, char **argv)
{
   printf("Generating matrix list!\n");
   if(argc<=2) {
      printf("Usage: %s <qtd_matrizes> <caminho_list_matrizes>\n", argv[0]);
      exit(1);
   }
   int n_mat = atoi(argv[1]);
   char* filename = argv[2];
   write_matrix_list(n_mat, filename, MATRIX_ORDER);
   return 0;
}
