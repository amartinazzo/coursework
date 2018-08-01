C = gcc
CFLAGS = -Wall -pedantic -O3 -march=native -fopenmp -pthread

# vê quais flags estão ativas para o processador nativo
# gcc -march=native -dM -E - < /dev/null | egrep "SSE|AVX" | sort

all:
	$(C) $(CFLAGS) -o main ep1.c
	./main o Q.txt Q.txt C.txt
	./main p Q.txt Q.txt C.txt

test:
	$(C) $(CFLAGS) -o main ep1.c
	./main p D.txt E.txt C.txt
	./main o D.txt E.txt C.txt
	./main s D.txt E.txt C.txt

clean:
	rm main
