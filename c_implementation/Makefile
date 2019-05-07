IDIR =./weights/network3
CC=gcc
CFLAGS=-I$(IDIR) -g -O0

LIBS=-lm -fopenmp

simple_mlp.o: simple_mlp.c
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) weights/network2/*.o


