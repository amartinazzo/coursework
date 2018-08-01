all:
	gcc -pthread -o ep2 ep2.c
	./ep2 5 5

run:
	./ep2 5 5

perf:
	perf stat -r 1000 ./ep2 5 5 0 > test_perf

clean:
	rm ep2
