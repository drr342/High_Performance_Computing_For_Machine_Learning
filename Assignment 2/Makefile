all: ./lab2  

CUDADIR :=  /share/apps/cuda/9.0.176
CUDNNDIR := /share/apps/cudnn/9.0v7.0.5

CPPFLAGS := -I$(CUDADIR)/include -I$(CUDNNDIR)/include
CFLAGS := -O2 --std=c++11

LDFLAGS := -L$(CUDADIR)/lib -L$(CUDNN)/lib
LDLIBS := -lcublas -lcudnn

NVCC := nvcc
CC := $(NVCC)

lab2.o: lab2.cu 

%.o: %.cu
	$(NVCC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm ./lab2
