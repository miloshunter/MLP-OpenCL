CONF?=network.conf

CC=gcc
CFLAGS=-I$(IDIR) -g -O0

LIBS=-lm -fopenmp


load_parameters.o: load_parameters.c
	$(CC) -c $^

simple_mlp.o: simple_mlp.c load_parameters.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

run_load: load_parameters.o
	./load_parameters.o $(CONF)

clean:
	@rm -f *.o
	@cd opencl_implementation && make clean
