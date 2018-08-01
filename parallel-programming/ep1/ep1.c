/**

EP1 - matrix multiplication

**/
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#define PTHREADS "p"
#define OPENMP "o"
#define __AVX__ 0       // flag que ativa AVX na compilação
#define NTHREADS 4      // quantidade de threads para OpenMP (= quantidade de cores)
#define BLOCKSIZE 512   // tamanho do partição da matriz (cabe no cache)
#define VSIZE 8         // tamanho dos vetores usados no produto interno
#define MMC 16          // os blocos das matrizes devem ser múltiplos de 2*VZISE

double **A, **B, **BT, **C;
size_t m, p, n;
size_t m_block, n_block, p_block;

struct timeval t0, t1;

typedef struct
{
    size_t i; size_t j; size_t k;
} blockdata;

void block_multiply();
void block_multiply_openmp();
void block_multiply_pthreads();
size_t fill(size_t n);
size_t get_block_size(size_t n);
void init_matrix(double ***matrix, size_t m, size_t n);
void *inner_loop(void *block_data);
size_t min(size_t a, size_t b);
void multiply();
void print_matrix(double **C, size_t m, size_t n);
void read_file(char *filename, double ***matrix, size_t *m, size_t *n, size_t *m_orig, size_t *n_orig);
void save_file(char *filename, double **matrix, size_t m, size_t n);
long time_elapsed (struct timeval t0, struct timeval t1);
void transpose(double ***matrix, double ***matrix_out, size_t m, size_t n);
void zeroes_matrix(double ***matrix, size_t m, size_t n);

int main(int argc, char *argv[])
{
    char *mode;
    size_t m_orig, n_orig, p_b, p_orig;

    if(argc != 5)
    {
        printf("parâmetros requeridos: main <implementação = p | o> <caminho_matr_A> <caminho_matr_B> <caminho_matr_C>\n");
        return 1;
    }

    // lê matrizes A e B, alocando memória e passando os valores dos arquivos para os pointers

    read_file(argv[2], &A, &m, &p, &m_orig, &p_orig);
    read_file(argv[3], &B, &p, &n, &p_b, &n_orig);

    if(p_orig != p_b)
    {
        printf("as dimensões das matrizes são incompatíveis\n");
        return 1;
    }

    // inicializa matrizes

    init_matrix(&C, m, n);
    init_matrix(&BT, n, p);

    transpose(&B, &BT, p, n);
    free(B);

    // define tamanhos de bloco

    m_block = get_block_size(m);
    n_block = get_block_size(n);
    p_block = get_block_size(p);

    // faz operações de multiplicação em bloco

    mode = argv[1];

    if(strcmp(mode, PTHREADS) == 0) {
        block_multiply_pthreads();
        print_matrix(C, m, n);
    } else if(strcmp(mode, OPENMP) == 0) {
        block_multiply_openmp();
        print_matrix(C, m, n);
    } else {
        // serial
        block_multiply();
        print_matrix(C, m, n);
    }

    // salva matriz C

    save_file(argv[4], C, m_orig, n_orig);

    return 0;

}


long time_elapsed (struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
}


size_t fill(size_t n)
{
    if(n % MMC != 0)
        return (n/MMC+1)*MMC;
    return n;
}


void read_file(char *filename, double ***matrix, size_t *m, size_t *n, size_t *m_orig, size_t *n_orig)
{
    FILE *fp;
    size_t i, j;
    double value;

    fp = fopen(filename, "r");
    fscanf(fp, "%lu %lu", m_orig, n_orig);

    *m = fill(*m_orig);
    *n = fill(*n_orig);

    printf("%s %lu x %lu\n", filename, *m_orig, *n_orig);

    *matrix = (double **) malloc(sizeof(double *)*(*m));
    for(i=0; i < *m; i++)
        (*matrix)[i] = (double *) calloc(*n, sizeof(double));

    while(fscanf(fp, "%lu %lu %lf", &i, &j, &value) != EOF) {
        (*matrix)[i-1][j-1] = value; // no arquivo de entrada, as posições começam de (1,1)
    }

    fclose(fp);
}


void save_file(char *filename, double **matrix, size_t m, size_t n)
{
    FILE *fp;
    fp = fopen(filename, "w");
    size_t i, j;
    double val;

    fprintf(fp, "%lu %lu\n", m, n);

    for(i=0; i < m; i++) {
        for(j=0; j < n; j++) {
            val = matrix[i][j];
            if(val != 0)
                fprintf(fp, "%lu %lu %lf\n", i+1, j+1, val);
        }
    }

    fclose(fp);
}


void init_matrix(double ***matrix, size_t m, size_t n)
{
    size_t i;
    *matrix = (double **) malloc(sizeof(double *)* m);
    for(i=0; i < m; i++)
        (*matrix)[i] = (double *) calloc(n, sizeof(double));
}


void zeroes_matrix(double ***matrix, size_t m, size_t n)
{
    size_t i, j;
    for(i=0; i < m; i++)
    {
        for(j=0; j < n; j++)
            (*matrix)[i][j] = 0.;
    }
}


void transpose(double ***matrix_in, double ***matrix_out, size_t m, size_t n) {
    size_t i, j;
    for(i=0; i< m; ++i) {
        for(j=0; j < n; ++j) {
            (*matrix_out)[j][i] = (*matrix_in)[i][j];
        }
    }
}


void print_matrix(double **C, size_t m, size_t n)
{
    size_t i,j;
    for(i=0; i < m; i++)
    {
        for(j=0; j < n; j++)
            printf("%lf ", C[i][j]);
        printf("\n");
    }
}


size_t get_block_size(size_t n) {
    if(n > BLOCKSIZE)
        return BLOCKSIZE;
    return n;
}


size_t min(size_t a, size_t b)
{
    if(a < b) {
        return a;
    } else {
        return b;
    }
}


double dot_prod(size_t i, size_t j, size_t k) {
    // faz as instruções vetoriais apenas se o compilador ativou a flag de avx
    if(__AVX__ == 1)
    {
        __m256d A_vec = _mm256_set_pd(A[i][k], A[i][k+1], A[i][k+2], A[i][k+3]);
        __m256d B_vec = _mm256_set_pd(BT[j][k], BT[j][k+1], BT[j][k+2], BT[j][k+3]);
        __m256d Ai_vec = _mm256_set_pd(A[i][k+4], A[i][k+5], A[i][k+6], A[i][k+7]);
        __m256d Bi_vec = _mm256_set_pd(BT[j][k+4], BT[j][k+5], BT[j][k+6], BT[j][k+7]);
        
        __m256d mul_vec = _mm256_mul_pd(A_vec, B_vec);
        __m256d mul_vec_1 = _mm256_mul_pd(Ai_vec, Bi_vec);
        __m256d temp = _mm256_hadd_pd(mul_vec, mul_vec_1);
        __m128d hi128 = _mm256_extractf128_pd(temp, 1);

        // vetor com duas posições: a primeira é A_vec dot B_vec e a segunda é Ai_vec dot Bi_vec
        __m128d dotproduct = _mm_add_pd(_mm256_castpd256_pd128(temp), hi128);
        
        return dotproduct[0] + dotproduct[1];

    // faz as instruções normais caso a flag esteja desativada (ocorre erro de instrução ilegal)
    } else {
        return A[i][k]*BT[j][k] + A[i][k+1]*BT[j][k+1] + A[i][k+2]*BT[j][k+2] + A[i][k+3]*BT[j][k+3] + 
            + A[i][k+4]*BT[j][k+4] + A[i][k+5]*BT[j][k+5] + A[i][k+6]*BT[j][k+6] + A[i][k+7]*BT[j][k+7];
    }
}


void multiply()
{
    double sum;
    size_t i, j, k;

    gettimeofday(&t0, NULL);

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            sum = 0.0;
            for (k = 0; k < p; k+=VSIZE)
            {
                sum += dot_prod(i, j, k);
            }
            C[i][j] += sum;
        }
    }

    gettimeofday(&t1, NULL);

    printf("\nmultiplicação sequencial\ntempo: %ld us\n\n", time_elapsed(t0, t1));
}


void block_multiply()
{
    size_t i, k, j, ii, jj, kk;
    double sum;

    gettimeofday(&t0, NULL);

    for (i=0; i < m; i += m_block)
    {
        for(j=0; j < n; j += n_block)
        {
            for (k=0; k < p; k += p_block)
            {
                for (ii=i; ii < min(i+m_block, m); ii++)
                {
                    for (jj=j; jj < min(j+n_block, n); jj++)
                    {
                        sum = 0.0;
                        for (kk=k; kk < min(k+p_block, p); kk+=VSIZE)
                            sum += dot_prod(ii, jj, kk);
                        C[ii][jj] += sum;
                    }
                }
            }
        }
    }

    gettimeofday(&t1, NULL);

    printf("\nmultiplicação em bloco sequencial\ntempo: %ld us\n\n", time_elapsed(t0, t1));
}


void *inner_loop(void *block_data)
{
    blockdata *data = (blockdata*) block_data;
    double sum;
    size_t ii, jj, kk, i, j, k;
    i = (*data).i; j = (*data).j;
    // printf("data inside block: i = %lu, j = %lu\n", (*data).i, (*data).j);

    for (k=0; k < p; k += p_block)
    {
        for (ii=i; ii < min(i + m_block, m); ii++)
        {
            for (jj=j; jj < min(j + n_block, n); jj++)
            {
                sum = 0.0;
                for (kk = k; kk < min(k + p_block, p); kk+=VSIZE)
                    sum += dot_prod(ii, jj, kk);;
                C[ii][jj] += sum;
            }
        }
    }

    // free(block_data);
    // return 0;
}


void block_multiply_pthreads()
{
    blockdata* blocks;
    pthread_t *threads;
    size_t i, j, k;
    size_t size = m/m_block * n/n_block * p/p_block;
    size_t id = 0;
    int error = 0;


    threads = (pthread_t *) malloc(sizeof(pthread_t)*size);
    blocks = (blockdata *) malloc(sizeof(blockdata)*size);
    
    if(threads == NULL)
    {
        printf("threads malloc falhou.");
        exit(1);
    }

    if(blocks == NULL)
    {
        printf("block malloc failed.\n");
        exit(1);
    }

    gettimeofday(&t0, NULL);

    for (i=0; i < m; i += m_block)
    {
        for(j=0; j < n; j += n_block)
        {
            blocks[id].i = i;
            blocks[id].j = j;
            // blocks[id].k = k;
            // printf("data before creating thread: id=%lu, i = %lu, j = %lu\n", id, blocks[id].i, blocks[id].j);
            error = pthread_create(&threads[id], NULL, inner_loop, (void *) &blocks[id]);
            if(error) {
                printf("pthread_create() returned %d\n", error);
                exit(1);
            }
            id++;
        }
    }

    for (i=0; i < id; i++)
        pthread_join(threads[i], NULL);

    gettimeofday(&t1, NULL);

    printf("\nmultiplicação em bloco com pthreads\ntempo: %ld us\n\n", time_elapsed(t0, t1));
}


void block_multiply_openmp()
{
    size_t i, j, k, ii, jj, kk;
    double sum;

    gettimeofday(&t0, NULL);

    for (i=0; i < m; i += m_block)
    {
        for(j=0; j < n; j += n_block)
        {            
            for (k=0; k < p; k += p_block)
            {
                #pragma omp parallel for num_threads(NTHREADS) private(jj, kk)
                for (ii=i; ii < min(i+m_block, m); ii++)
                {
                    for (jj=j; jj < min(j+n_block, n); jj++)
                    {
                        sum = 0.0;
                        for (kk=k; kk < min(k+p_block, p); kk+=VSIZE)
                            sum += dot_prod(ii, jj, kk);
                        C[ii][jj] += sum;
                    }
                }
            }
        }
    }

    gettimeofday(&t1, NULL);

    printf("\nmultiplicação em bloco com openmp\ntempo: %ld us\n\n", time_elapsed(t0, t1));
}

