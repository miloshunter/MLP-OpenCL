IDIR =~/NVIDIA\ GPU\ Computing\ SDK/OpenCL/common/inc/
CC=gcc
CFLAGS=-I$(IDIR) -g -O0

ODIR=obj
LDIR =../lib

LIBS=-lm -lOpenCL

_DEPS = opencl_mlp.o
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

OBJ = opencl_mlp.o


$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

opencl_mlp: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) ../weights/network3/*.o opencl_utils.c ../weights/slike.o

opencl_utils.o:
	gcc -c opencl_utils.c

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 


 #-o main opencl_mlp.c opencl_utils.c -lOpenCL -g -O0 ../weights/*.c ../weights/network3/*.c -lm


# gcc -o test test_kernel.c opencl_utils.o -I ~/NVIDIA\ GPU\ Computing\ SDK/OpenCL/common/inc/ -lm -lOpenCL