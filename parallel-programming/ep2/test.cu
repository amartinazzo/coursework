#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "functions.h"
#include "reduction_cuda.h"
#include "reduction_seq.h"

#define MATRIX_ORDER 3

int main(int argc, char *argv[])
{
  int result;
  const char* filename;
  const char* description;
  int *y_cuda;
  int *y_seq ;

  /* Teste 1: Uma matriz */
  description = "Uma matriz";
  //given
  filename = "data/teste_1.txt";
  write_matrix_list(1, filename, MATRIX_ORDER);
  //when
  y_cuda = reduction_cuda(filename, MATRIX_ORDER);
  y_seq = reduction_seq(filename, MATRIX_ORDER);
  //then
  result = assert_vector(y_cuda, y_seq, MATRIX_ORDER*MATRIX_ORDER);
  print_test_result(description, result);

  /* Teste 2: Duas matrizes */
  description = "Duas matrizes";
  //given
  filename = "data/teste_2.txt";
  write_matrix_list(2, filename, MATRIX_ORDER);
  //when
  y_cuda = reduction_cuda(filename, MATRIX_ORDER);
  y_seq = reduction_seq(filename, MATRIX_ORDER);
  //then
  result = assert_vector(y_cuda, y_seq, MATRIX_ORDER*MATRIX_ORDER);
  print_test_result(description, result);

  /* Teste 3: Quantidade ímpar de matrizes */
  description = "Quantidade ímpar de matrizes";
  //given
  filename = "data/teste_impar.txt";
  write_matrix_list(3, filename, MATRIX_ORDER);
  //when
  y_cuda = reduction_cuda(filename, MATRIX_ORDER);
  y_seq = reduction_seq(filename, MATRIX_ORDER);
  //then
  result = assert_vector(y_cuda, y_seq, MATRIX_ORDER*MATRIX_ORDER);
  print_test_result(description, result);


  /* Teste 4: Quantidade par de matrizes */
  description = "Quantidade par de matrizes";
  //given
  filename = "data/teste_par.txt";
  write_matrix_list(4, filename, MATRIX_ORDER);
  //when
  y_cuda = reduction_cuda(filename, MATRIX_ORDER);
  y_seq = reduction_seq(filename, MATRIX_ORDER);
  //then
  result = assert_vector(y_cuda, y_seq, MATRIX_ORDER*MATRIX_ORDER);
  print_test_result(description, result);

  /* Teste 5: 10 matrizes */
  description = "Dez matrizes";
  //given
  filename = "data/teste_10.txt";
  write_matrix_list(10, filename, MATRIX_ORDER);
  //when
  y_cuda = reduction_cuda(filename, MATRIX_ORDER);
  y_seq = reduction_seq(filename, MATRIX_ORDER);
  //then
  result = assert_vector(y_cuda, y_seq, MATRIX_ORDER*MATRIX_ORDER);
  print_test_result(description, result);

  /* Teste 6: 255 matrizes */
  description = "255 matrizes (n de threads por bloco - 1)";
  //given
  filename = "data/teste_255.txt";
  write_matrix_list(255, filename, MATRIX_ORDER);
  //when
  y_cuda = reduction_cuda(filename, MATRIX_ORDER);
  y_seq = reduction_seq(filename, MATRIX_ORDER);
  //then
  result = assert_vector(y_cuda, y_seq, MATRIX_ORDER*MATRIX_ORDER);
  print_test_result(description, result);

  /* Teste 7: 256 matrizes */
  description = "256 matrizes (n de threads por bloco)";
  //given
  filename = "data/teste_256.txt";
  write_matrix_list(256, filename, MATRIX_ORDER);
  //when
  y_cuda = reduction_cuda(filename, MATRIX_ORDER);
  y_seq = reduction_seq(filename, MATRIX_ORDER);
  //then
  result = assert_vector(y_cuda, y_seq, MATRIX_ORDER*MATRIX_ORDER);
  print_test_result(description, result);

  /* Teste 8: 257 matrizes */
  description = "257 matrizes (n de threads por bloco + 1)";
  //given
  filename = "data/teste_257.txt";
  write_matrix_list(257, filename, MATRIX_ORDER);
  //when
  y_cuda = reduction_cuda(filename, MATRIX_ORDER);
  y_seq = reduction_seq(filename, MATRIX_ORDER);
  //then
  result = assert_vector(y_cuda, y_seq, MATRIX_ORDER*MATRIX_ORDER);
  print_test_result(description, result);

  /* Teste 9: 10k matrizes */
  description = "10k matrizes (usa bastante memória da gpu)";
  //given
  filename = "data/teste_10k.txt";
  write_matrix_list(10000, filename, MATRIX_ORDER);
  //when
  y_cuda = reduction_cuda(filename, MATRIX_ORDER);
  y_seq = reduction_seq(filename, MATRIX_ORDER);
  //then
  result = assert_vector(y_cuda, y_seq, MATRIX_ORDER*MATRIX_ORDER);
  print_test_result(description, result);

  /* Teste 9: 100k matrizes */
  description = "100k matrizes (usa bastante memória da gpu)";
  //given
  filename = "data/teste_100k.txt";
  write_matrix_list(100000, filename, MATRIX_ORDER);
  //when
  y_cuda = reduction_cuda(filename, MATRIX_ORDER);
  y_seq = reduction_seq(filename, MATRIX_ORDER);
  //then
  result = assert_vector(y_cuda, y_seq, MATRIX_ORDER*MATRIX_ORDER);
  print_test_result(description, result);

  /* Teste 11: 1M matrizes */
  description = "1M matrizes (usa bastante memória da gpu)";
  //given
  filename = "data/teste_1M.txt";
  write_matrix_list(1000000, filename, MATRIX_ORDER);
  //when
  y_cuda = reduction_cuda(filename, MATRIX_ORDER);
  y_seq = reduction_seq(filename, MATRIX_ORDER);
  //then
  result = assert_vector(y_cuda, y_seq, MATRIX_ORDER*MATRIX_ORDER);
  print_test_result(description, result);

  /* Teste 12: 10M matrizes */
  description = "10M matrizes (usa bastante memória da gpu)";
  //given
  filename = "data/teste_10M.txt";
  write_matrix_list(10000000, filename, MATRIX_ORDER);
  //when
  y_cuda = reduction_cuda(filename, MATRIX_ORDER);
  y_seq = reduction_seq(filename, MATRIX_ORDER);
  //then
  result = assert_vector(y_cuda, y_seq, MATRIX_ORDER*MATRIX_ORDER);
  print_test_result(description, result);
  return 0;
}
