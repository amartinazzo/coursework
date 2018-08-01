CC      = nvcc
CFLAGS  = -Xptxas --opt-level=3 -arch sm_30

main: main.cu reduction_cuda.cu reduction_seq.cu functions.cu
	$(CC) $(CFLAGS) -o $@ $^

generate_matrix_list: generate_matrix_list.cu functions.cu
	$(CC) $(CFLAGS) -o $@ $^

test: test.cu reduction_cuda.cu reduction_seq.cu functions.cu
	$(CC) $(CFLAGS) -o $@ $^

test_performance: test_performance.cu reduction_cuda.cu reduction_seq.cu functions.cu
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o main test test_performance generate_matrix_list
