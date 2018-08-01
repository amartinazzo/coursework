all:
	gcc -o mini-ep1-main mini-ep1.c
	./mini-ep1-main

clean:
	rm mini-ep1-main

compile:
	gcc -o mini-ep1-main mini-ep1.c

perf:
	perf stat -r 10 -B -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores ./mini-ep1-main 0
	perf stat -r 10 -B -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores ./mini-ep1-main 1
