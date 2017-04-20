CUDA_ROOT = /usr/local/cuda
ARMADILLO_ROOT = /usr
PCA_ROOT = /usr/local

INC =-Ivl/

#INC +=-I/usr/local/include 
#INC +=-I/usr/local/openssl/include/ -I/usr/lib/x86_64-linux-gnu/gcc/x86_64-linux-gnu/4.5/include 
INC +=-I./
LIB     = -lm -lrt -lpthread -ldl -std=c++0x
LIB     += -L$(CUDA_ROOT)/lib64/ -lcudart -lcufft 
LIB     += -fopenmp -lcublas

CXX = g++ -pipe

NVCC	= $(CUDA_ROOT)/bin/nvcc -arch=sm_35 -m 64 $(INC)

CXX = g++ -pipe
CC = gcc -pipe
LINK	= -O3 -pthread 

test: test.o cuda_files.o
	$(CXX) $(LINK) -o test test.o cuda_files.o $(LIB)

cuda_files.o test.o: cuda_files.cu test.cu
	$(NVCC) -I$(CUDA_ROOT)/include/ -O3 -c cuda_files.cu test.cu -lcublas -std=c++11


.DEFAULT:
	@echo The target \"$@\" does not exist in Makefile.

