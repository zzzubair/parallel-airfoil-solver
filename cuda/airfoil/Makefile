NVCC = nvcc
CC = gcc
CFLAGS = -O3
CUDAFLAGS = -O3 -arch=sm_80 --use_fast_math
LIBFLAGS = -lm

OBJDIR = obj

_OBJ_CUDA = args.o data.o setup.o vtk.o boundary.o
OBJ_CUDA = $(patsubst %,$(OBJDIR)/%,$(_OBJ_CUDA))

_OBJ_SERIAL = args.o data.o setup.o vtk.o boundary.o airfoil.o
OBJ_SERIAL = $(patsubst %,$(OBJDIR)/%,$(_OBJ_SERIAL))

.PHONY: directories all clean

all: directories airfoil_cuda airfoil

airfoil_cuda: directories $(OBJ_CUDA) obj/airfoil_cuda.o
	$(NVCC) -o $@ $(OBJ_CUDA) obj/airfoil_cuda.o $(CUDAFLAGS) $(LIBFLAGS)

obj/airfoil_cuda.o: airfoil_cuda.cu | directories
	$(NVCC) -c -o $@ $< $(CUDAFLAGS)

airfoil: directories $(OBJ_SERIAL)
	$(CC) -o $@ $(OBJ_SERIAL) $(CFLAGS) $(LIBFLAGS)

obj/%.o: %.c | directories
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -rf $(OBJDIR)
	rm -f airfoil airfoil_cuda

directories: $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)
