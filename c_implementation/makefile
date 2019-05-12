CONF?=mlp.conf
IMG?=4.png

CC=gcc
CFLAGS=-I$(IDIR) -g -O0

LIBS=-lm -fopenmp -lpng

all: simple_mlp.o
	./simple_mlp.o $(CONF) $(IMG)

load_parameters.o: load_parameters.c load_parameters.h
	$(CC) -c $^

simple_mlp.o: simple_mlp.c load_parameters.o read_image.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

run_load: load_parameters.o
	./load_parameters.o $(CONF)

read_image.o: read_image.c
	gcc -c $^ -lpng

clean:
	@rm -f *.o
	@rm -f *.gch
	@cd opencl_implementation && make clean
