SHELL := /bin/bash
all: start compile execute clear
	rm *.o

start:
	rm -f *.output

compile:
	gcc -o cpu_parallel.o cpu_parallel/main.c -fopenmp -lm  
	python3 gerador_input/criar_input.py
	gcc -o cpu_normal.o cpu/main.c -lm
	nvcc -o gpu_otimizada.o gpu_otimizada/main.cu
	nvcc -o gpu_simples.o gpu_simples/main.cu
        # nvcc -o gpu_simples.o gpu_simples/main.cu
        # nvcc -o gpu_otimizado.o gpu_otimizado/main.cu

execute:
	# for number in {1..5} ;do  ./gpu_otimizada.o; done;
	for number in {1..5} ;do  ./gpu_simples.o; done;
	
	# for number in {1..5} ;do  ./cpu_normal.o; done;
	# for number in {1..5} ;do  ./cpu_parallel.o; done;
        # for number in {1..20} ;do  ./gpu_otimizado.o; done;


clear:
	rm *.o