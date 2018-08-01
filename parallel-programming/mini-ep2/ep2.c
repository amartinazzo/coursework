/**

Mini EP2 - frog puzzle

**/

//0,014215020 seconds time elapsed  

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define EMPTY -99

pthread_t *frogs;
pthread_t *manager;
pthread_mutex_t lock;

clock_t start, end;
double cpu_time_used;

int M, N;
int count_max = 10000;
int global_count = 0;
int lake_check = 1;
int *lake;
int verbose = 0;

long length;

void print_lake() {
    for(int i=0; i < length+1; i++) {
        printf("%d ", lake[i]);
    }
    printf("\n");
}

void lake_init() {
    int i;
    lake = malloc(sizeof(int)*(length+1));
    lake[M] = EMPTY;
    for(i=0; i < M; i++) lake[i] = i;
    for(i=length+1; i > N; i--) lake[i] = i-1;
    if(verbose) print_lake();
}

void update_lake(int pos, int delta_pos, long frog_id) {
    lake[pos] = EMPTY;
    pos = pos + delta_pos;
    lake[pos] = frog_id;
    global_count = 0;
    if(verbose) {
        printf("%ld pulou:\n", frog_id);
        print_lake();
    }
}

void *check_lake(void *t) {
    long length = (long) t;
    int i, deadlock;
    int *lake_copy;

    lake_copy = malloc(sizeof(int)*(length+1));
    
    while(lake_check) {
        deadlock = 1;
        pthread_mutex_lock(&lock);
        for(i=0; i < length+1; i++) lake_copy[i] = lake[i];
        pthread_mutex_unlock(&lock);
        for(i=0; i < length+1; i++) {
            if ( (lake_copy[i] < M && i < length+1 && (lake_copy[i+2] == EMPTY || lake_copy[i+1] == EMPTY)) || 
                (lake_copy[i] > M && i>0 && (lake_copy[i-2] == EMPTY || lake_copy[i-1] == EMPTY)) ) {
                    deadlock = 0;
                    break;
                }
        }
        if(deadlock) {
            printf("árbitro detectou deadlock.\n");
            lake_check = 0;
        }
    }

    pthread_exit(NULL);
}

void *jump(void *t){
    long frog_id = (long) t;
    int pos = frog_id < M ? frog_id : (frog_id+1);

    while(global_count < count_max && lake_check) {

        // mover sapo
        if(frog_id < M) {
            pthread_mutex_lock(&lock);
            if(verbose) printf("sapo %ld tentando pular...\n", frog_id);
            if(pos == length+1) {
                global_count++;
            } else if (lake[pos+2] == EMPTY) {
                update_lake(pos, 2, frog_id);
            } else if(lake[pos+1] == EMPTY) {
                update_lake(pos, 1, frog_id);
            } else {
                global_count++;
            }
            pthread_mutex_unlock(&lock);

        // mover sapa
        } else if(frog_id > M) {
            pthread_mutex_lock(&lock);
            if(verbose) printf("sapa %ld tentando pular...\n", frog_id);
            if(pos == 0) {
                global_count++;
            } else if (lake[pos-2] == EMPTY) {
                update_lake(pos, -2, frog_id);
            } else if(lake[pos-1] == EMPTY) {
                update_lake(pos, -1, frog_id);
            } else {
                global_count++;
            }
            pthread_mutex_unlock(&lock);
        }
    }

    pthread_exit(NULL);
};

int main(int argc, char *argv[]) {
    int error;
    long t;

    // recebe parâmetros de entrada

    if(argc < 3) {
        printf("é necessário fornecer os parâmetros M (números de sapos) e N (número de rãs).");
        exit(1);
    }

    M = atoi(argv[1]);
    N = atoi(argv[2]);

    if(M == 0 || N == 0) {
        printf("os valores fornecidos para M e/ou N são inválidos.");
        exit(1);
    }

    if(argv[3] != NULL) verbose = atoi(argv[3]);

    length = M+N;
    count_max = count_max*length;
    frogs = malloc(sizeof(pthread_t)*(length+1));
    manager = malloc(sizeof(pthread_t));

    if(frogs == NULL || manager == NULL) {
        printf("threads malloc failed\n");
        exit(1);
    }


    // inicializa estado da lagoa com id dos sapos nas posições iniciais

    lake_init();

    // inicializa mutex

    if(pthread_mutex_init(&lock, NULL) != 0) {
        printf("mutex init failed\n");
        exit(1);
    }

    // cria threads (frogs)

    start = clock();

    for(t=0; t < length; t++) {
        error = pthread_create(&frogs[t], NULL, jump, (void *) t);
        if(error) {
            printf("pthread_create() returned %d\n", error);
            exit(1);
        }
    }

    // cria uma thread adicional (árbitro/gerenciadora)

    error = pthread_create(manager, NULL, check_lake, (void *) length);
        if(error) {
            printf("pthread_create() returned %d\n", error);
            exit(1);
        }

    // espera todas as threads de animais finalizarem para terminar o programa

    for (t=0; t < length; t++) {
        pthread_join(frogs[t], NULL);  
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    if(global_count >= count_max) printf("o limite do contador foi atingido.\n");
    printf("tempo gasto: %0.2fs\n", cpu_time_used);
    printf("contador: %d\n", global_count);
    printf("estado final: ");
    print_lake();

    return 0;
}
