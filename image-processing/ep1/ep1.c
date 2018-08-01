/**

EP1 - estimando o nível de ruído em imagens .pgm

Este programa recebe como parâmetros um arquivo de imagem .pgm e um tamanho de janela, e retorna a variância 
mínima de brilho para essa janela e o tempo de execução para cada uma das três formulações:

var_1 = 1/n * somatório( (x_i - x_média)^2 )
var_2 = 1/n * [ somatório(x_i^2) - n*x_média^2 ]
var_3 = 1/n * [ somatório(x_i^2) - n*x_média^2 ] com imagem integral

Para validar o programa, foram usadas imagens dos seguintes bancos:
- imagens originais: http://neelj.com/projects/twocolordeconvolution/supplemental_results/orig/
- imagens com ruído de 10%: http://neelj.com/projects/twocolordeconvolution/supplemental_results/noisy/

As imagens podem ser convertidas para o formato .pgm (tipo p2, não comprimido) usando o seguinte comando do 
imagemagick:

convert arquivo-de-entrada.jpg -compress none arquivo-de-saida.pgm

Um exemplo de teste foi feito com duas imagem de ursos (disponíveis nos bancos de imagem citados acima), 
uma na versão original (bears.jpg) e outra com ruído (bears_0.10_noisy.jpg). Adotando uma janela T=10, a 
variância mínima é 9 na imagem original e 180 na imagem com ruído. O programa também foi testado com o 
pacote .zip que acompanha o enunciado, contendo cinco imagens com níveis visivelmente crescentes de ruído. 
Para uma janela T=10, as variâncias mínimas encontradas foram: 1, 11, 39, 85, 147.

Observou-se que a variância mínima aumenta com o tamanho de janela adotado, conforme o esperado, pois pedaços 
maiores de imagens são menos homogêneos. Para uma janela suficientemente grande (T=200), até mesmo a imagem 
que não apresenta ruído visível (fig01.pgm) apresentou uma variância alta, de 684. Janelas maiores capturam 
variações no brilho da própria imagem ao invés das variações granulares correspondentes ao ruído. Portanto, 
para estimar o nível de ruído, não é vantajoso escolher janelas maiores.

Verificou-se que o tempo de execução de var_3, que usa imagens integrais, é o menor dos três. Mesmo para 
imagens pequenas (a partir de ~15 x 15), o custo adicional de calcular as imagens integrais é compensado pela 
otimização da quantidade de acessos à matriz. Ao aumentar o tamanho da janela, o tempo de var_3 diminui, pois 
ainda menos acessos à imagem são necessários; já os tempos de var_1 e var_2 chegam a aumentar duas ordens de 
grandeza, pois é preciso iterar ao longo de janelas maiores.

**/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

clock_t start, end;
double cpu_time_used;

int verbose = 0;

/**

cabeçalho de funções

**/

int area(long **integral_img, int i, int j, int side);
void calc_performance(int variance_func(int **, int, int, int, int, int, long **, long **), int **img, int ksize,
	int M, int N, int **squared_img, long **integral_img, long **integral_img_squared);
int compare_int(const void *a, const void *b);
void init_integral_image(int **img, long **integral_img, int M, int N);
void squared_matrix(int **mat, int **mat_squared, int M, int N);
int variance_1(int **img, int ksize, int i, int j, int M, int N, long **mat1, long **mat2);
int variance_2(int **img, int ksize, int i, int j, int M, int N, long **mat1, long **mat2);
void init_variance_3(int M, int N, long **integral_img, long **integral_img_squared, int **img, int **squared_img);
int variance_3(int **img, int ksize, int i, int j, int M, int N, long **integral_img, long **integral_img_squared);
int min_variance(int variance_func(int **, int, int, int, int, int, long **, long **), int **img, int ksize,
	int M, int N, long **integral_img, long **integral_img_squared);


/**

main

**/

int main(int argc, char **argv) {

	if(argc < 3) {
		printf("dois parâmetros são requeridos: o nome do arquivo pgm e o tamanho da janela.\n");
		return 1;
	}

	// inicializa variáveis

	FILE *fp;
	char *imagefile;
	char filetype;
    int M, N, i, j, ksize;
    int maxvalue;
	int  **img;
	int **squared_img;
	long **integral_img;
	long **integral_img_squared;


	// lê parâmetros e arquivo de imagem

    imagefile = argv[1];
    ksize = atoi(argv[2]);
    if(argv[3] != NULL) verbose = atoi(argv[3]);
    fp = fopen (imagefile,"r");
    printf("arquivo: %s\n", imagefile);
    printf("tamanho da janela: %d\n", ksize);
    fscanf(fp, "%s %d %d %d", &filetype, &M, &N, &maxvalue); // lê cabecalho do arquivo pgm
    printf("tamanho da imagem: %d x %d\n", M, N);
    printf("brilho máximo: %d\n", maxvalue);


    // aloca matrizes

    img = (int **)malloc(sizeof(int *)*M);
	squared_img = (int **)malloc(sizeof(int *)*M);
	integral_img = (long **)malloc(sizeof(long *)*M);
	integral_img_squared = (long **)malloc(sizeof(long *)*M);
    
	for(i=0; i < M; i++) {
		img[i] = (int *)malloc(sizeof(int)*N);
		squared_img[i] = (int *)malloc(sizeof(int)*N);
  		integral_img[i] = (long *)malloc(sizeof(long)*N);
  		integral_img_squared[i] = (long *)malloc(sizeof(long)*N);
	}


    // lê elementos da matriz

	if(verbose==1) printf("imagem original:\n\n");
   	for(i=0; i < M; i++) {
    	for(j=0; j < N; j++) {
    		fscanf(fp, "%d", &img[i][j]);
    		if(verbose==1) printf("%d ", img[i][j]);
    	}
    	if(verbose==1) printf("\n");
    }


    // calcula variância mínima em cada um dos métodos

    // método 1
    printf("\nmétodo 1:");
    calc_performance(variance_1, img, ksize, M, N, NULL, NULL, NULL);

    // método 2
    printf("\nmétodo 2:");
    calc_performance(variance_2, img, ksize, M, N, NULL, NULL, NULL);  

    //método 3
	printf("\nmétodo 3:");
    calc_performance(variance_3, img, ksize, M, N, squared_img, integral_img, integral_img_squared);

	return 0;
}

/**

funções

**/

int compare_int(const void *a, const void *b) {
    int *x = (int *) a;
    int *y = (int *) b;
    return *x - *y;
}

void init_integral_image(int **img, long **integral_img, int M, int N) {
	int i, j;
	if(verbose==1) printf("\nintegral image:\n");
	for(i=0; i < M; i++) {
		for(j=0; j < N; j++) {
			if(i==0 && j==0) {
				integral_img[i][j] = img[i][j];
			} else if(i==0) {
				integral_img[i][j] = img[i][j] + integral_img[i][j-1];
			} else if(j==0) {
				integral_img[i][j] = img[i][j] + integral_img[i-1][j];
			} else {
				integral_img[i][j] = img[i][j] + integral_img[i-1][j] + integral_img[i][j-1] - 
				integral_img[i-1][j-1];
			}
			if(verbose==2) printf("%ld ", integral_img[i][j]);
		}
		if(verbose==2) printf("\n");
	}
}

// método para calcular o quadrado da matriz elemento a elemento
void squared_matrix(int **mat, int **mat_squared, int M, int N) {
	if(verbose==1) printf("\nsquared matrix:\n");
	for(int i=0; i<M; i++) {
		for(int j=0; j<N; j++) {
			mat_squared[i][j] = mat[i][j] * mat[i][j];
			if(verbose==2) printf("%d ", mat_squared[i][j]);
		}
		if(verbose==2) printf("\n");
	}
}

// método para calcular uma área quadrada na imagem integral, dados o ponto inicial e o tamanho da aresta (janela)
int area(long **integral_img, int i, int j, int side) {
	int area;
	int L = side - 1;
	if(i==0 && j==0) {
		area = integral_img[L][L];
	} else if(i==0) {
		area = integral_img[L][j+L] - integral_img[L][j-1];
	} else if(j==0) {
		area = integral_img[i+L][L] - integral_img[i-1][L];
	} else {
		area = integral_img[i+L][j+L] - integral_img[i-1][j+L] - integral_img[i+L][j-1] + integral_img[i-1][j-1];
	}
	return area;
}


// método (1) para calcular a variância: pela eq. 2 ("clássica")
int variance_1(int **img, int ksize, int i, int j, int M, int N, long **mat1, long **mat2) {
	int a, b;
	double sum = 0;
	double mean;

	for(a=i; a<ksize+i; a++) {
		for(b=j; b<ksize+j; b++) {
			sum += img[a][b];
		}
	}

	mean = sum/(ksize*ksize);
	sum = 0;

	for(a=i; a<ksize+i; a++) {
		for(b=j; b<ksize+j; b++) {
			sum += (img[a][b] - mean) * (img[a][b] - mean);
		}
	}

	return sum/(ksize*ksize);
}


// método (2) para calcular a variância: pela eq. 4 (alternativa) sem imagem integral
int variance_2(int **img, int ksize, int i, int j, int M, int N, long **mat1, long **mat2) {
	int a, b;
	double squared_sum = 0;
	double sum = 0;
	int n = ksize*ksize;

	for(a=i; a<ksize+i; a++) {
		for(b=j; b<ksize+j; b++) {
			sum += img[a][b];
			squared_sum += img[a][b] * img[a][b];
		}
	}

	return (squared_sum/n - sum*sum/(n*n));
}

// método para inicializar as imagens integrais que serão usadas para cálculo da variância
void init_variance_3(int M, int N, long **integral_img, long **integral_img_squared, int **img, int **squared_img) {
	init_integral_image(img, integral_img, M, N);
	squared_matrix(img, squared_img, M, N);
	init_integral_image(squared_img, integral_img_squared, M, N);
	free(squared_img);
	squared_img = NULL;
}

// método (3) para calcular a variância: pela eq. 4 (alternativa) COM imagem integral
int variance_3(int **img, int ksize, int i, int j, int M, int N, long **integral_img, long **integral_img_squared) {
	int A_squared = area(integral_img_squared, i, j, ksize);
	int A = area(integral_img, i, j, ksize);
	double n = ksize*ksize;
	double var1 = A/n;
	return A_squared/n - var1*var1;
}


// método para encontrar a menor variância
int min_variance(int variance_func(int **, int, int, int, int, int, long **, long **), int **img,
	int ksize, int M, int N, long **integral_img, long **integral_img_squared) {
	int i, j, k;
	int m = M+1-ksize;
	int n = N+1-ksize;
	int *var_array;
	var_array = (int *)malloc(sizeof(int)*m*n);

	if(verbose==1) printf("\nvetor de variâncias: ");

	for(i=0; i < m; i++) {
		for(j=0; j < n; j++) {
			k = j + i*n;
			var_array[k] = variance_func(img, ksize, i, j, M, N, integral_img, integral_img_squared);
			if(verbose==1) printf("%d ", var_array[k]);
		}
	}
	qsort(var_array, m*n, sizeof(int), compare_int);

	return var_array[0];
}

// método para estimar o tempo de execução dos cálculos de variância
void calc_performance(int variance_func(int **, int, int, int, int, int, long **, long **), int **img, int ksize,
	int M, int N, int **squared_img, long **integral_img, long **integral_img_squared) {
	start = clock();
	if(integral_img != NULL) init_variance_3(M, N, integral_img, integral_img_squared, img, squared_img);
	int min = min_variance(variance_func, img, ksize, M, N, integral_img, integral_img_squared); 
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("\nvariância mínima: %d", min);
	printf("\ntempo: %.1e\n", cpu_time_used);
}
