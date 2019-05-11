CC=gcc
CFLAGS=-I$(IDIR) -g -O0

LIBS=-lm -fopenmp


load_parameters.o: load_parameters.c
	$(CC) -o $@ $^

simple_mlp.o: simple_mlp.c
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) weights/network3/*.o  weights/*.o

run_load: load_parameters.o
	./load_parameters.o network.conf

clean:
	@rm -f *.o
	@cd opencl_implementation && make clean
