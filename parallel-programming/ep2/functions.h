void print_matrix(int** matrix, int n_rows, int n_cols);
void print_vector(int* vector, int D);
int get_n_mat(const char filename[]);
int assert_vector(int* a, int* b, int size);
void print_test_result(const char description[], int result);
int randMToN(int M, int N);
void write_matrix_list(int n_mat, const char filename[], int matrix_order);
long time_elapsed (struct timeval t0, struct timeval t1);
void print_performance_test_result(const char* description, long time_elapsed_cuda, long time_elapsed_seq);
