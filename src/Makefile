CXXSRCS	= $(wildcard *.cpp)
#CSRCS	= $(wildcard ./gpuFV/vl/*.c)
CHEADERS	= $(wildcard *.h)
#CXXHEADERS = $(wildcard ./gpuFV/vl/*.h)
CXXDIR	= $(notdir $(CXXSRCS))
#CDIR = $(notdir $(CSRCS))
CXXOBJECTS	= $(patsubst %.cpp,%.o, $(CXXDIR)) 
#COBJECTS	= $(patsubst %.c,%.o, $(CDIR)) 

CUDA_ROOT = /usr/local/cuda
ARMADILLO_ROOT = /usr
PCA_ROOT = /usr/local

#INC +=-I/usr/local/include 
#INC +=-I/usr/local/openssl/include/ -I/usr/lib/x86_64-linux-gnu/gcc/x86_64-linux-gnu/4.5/include 
INC += -I./
LIB  = -lm -lrt -lpthread -ldl -std=c++0x
LIB += -L$(CUDA_ROOT)/lib64/ -lcudart -lcufft 
LIB += -fopenmp -lcublas
LIB += -lvl -lalg -L./alglib -L./vl `pkg-config --libs opencv-2.4.9 libavformat libavdevice libavutil libavcodec libswscale`

CXX = g++ -pipe

NVCC	= $(CUDA_ROOT)/bin/nvcc -arch=sm_35 -m 64 $(INC)

CXX = g++ -pipe
CC = gcc -pipe
LINK	= -O3 -pthread
CFLAGS	= -O3 -c -pthread -fopenmp -DVL_DISABLE_AVX
CXXFLAGS= -O3 -c -pthread -std=c++0x -fopenmp -DVL_DISABLE_AVX `pkg-config --cflags opencv-2.4.9`

compute_fv_gpu: $(CXXOBJECTS) cuda_files.o
	$(CXX) $(LINK) -o compute_fv_gpu $(CXXOBJECTS) cuda_files.o $(LIB)

cuda_files.o: cuda_files.cu 
	$(NVCC) -I$(CUDA_ROOT)/include/ -O3 -c cuda_files.cu -lcublas

#$(COBJECTS): $(CSRCS) $(CHEADERS) Makefile
#	$(CC) $(INC) $(CFLAGS) $(CSRCS)

$(CXXOBJECTS): $(CXXSRCS) $(CXXHEADERS) Makefile
	$(CXX) $(INC) $(CXXFLAGS) $(CXXSRCS)


#alglib/libalg.so: alglib/*.cpp
#	g++ -O2 -o alglib/libalg.so alglib/*.cpp -shared -fPIC -std=c++0x

.DEFAULT:
	@echo The target \"$@\" does not exist in Makefile.
