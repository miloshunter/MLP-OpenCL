CC=gcc
CFLAGS=-I$(IDIR) -g -O0

ODIR=obj
LDIR =../

LIBS=-lm -lOpenCL -lpng

_DEPS = opencl_mlp.o
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

OBJ = opencl_mlp.o

all: opencl_utils.o opencl_mlp

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

opencl_mlp: $(OBJ) ../load_parameters.o ../read_image.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) opencl_utils.c

opencl_utils.o:
	gcc -c opencl_utils.c

.PHONY: clean

clean:
	@rm -f *.o 
	@rm -f test
	@rm -f opencl_mlp.o
	@rm -f opencl_mlp

 #-o main opencl_mlp.c opencl_utils.c -lOpenCL -g -O0 ../weights/*.c ../weights/network3/*.c -lm


# gcc -o test test_kernel.c opencl_utils.o -I ~/NVIDIA\ GPU\ Computing\ SDK/OpenCL/common/inc/ -lm -lOpenCL
