all:
	gcc -o ep1 ep1.c
	./ep1 fig01.pgm 10

test bear:
	@echo "\n\nnoisy bear, T=10\n\n"
	./ep1 bear-noisy.pgm 10
	@echo "\n\nnoisy bear, T=100\n\n"
	./ep1 bear-noisy.pgm 100
	@echo "\n\nbear, T=10\n\n"
	./ep1 bear.pgm 10
	@echo "\n\nbear, T=100\n\n"
	./ep1 bear.pgm 100


clean:
	rm ep1