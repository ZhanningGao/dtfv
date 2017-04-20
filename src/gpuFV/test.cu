#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <ctime>
#include <set>
#include <iterator>
#include <cassert>
#include <dirent.h>
#include <random>

//#include "CL/cl.h"
//#include "Timer.h"
#include "cuda_files.h"

#include <cstring>
//#include "SDKFile.hpp"
extern "C"{
#include "vl/generic.h"
#include "vl/gmm.h"
#include "vl/fisher.h"
#include "vl/mathop.h"
}

#define random(x) ((rand()%x)/(float)x)
#define IDX2C(r,c, ROW_NUM) (c*ROW_NUM + r) //column frist


int main(int argc, char* argv[])
{
	printf("%s\n", "hello world");
	srand((int)time(0));

	int num_data = 10000;
	int dim_pca = 1024;
	int dim_ori = 2048;

	float *project = new float[dim_pca*dim_ori];
	float *mean = new float[dim_ori];
	float *data = new float[dim_ori*num_data];
	float *dst = new float[dim_pca*num_data];

	// init data
	for (int c = 0; c < num_data; c++){
		for (int r = 0; r < dim_ori; r++){
			data[IDX2C(r,c,dim_ori)] = random(10000);
		}
	}
	// init project
	for (int c = 0; c < dim_ori; c++){
		for (int r = 0; r < dim_pca; r++){
			project[IDX2C(r,c,dim_pca)] = random(10000);
		}
	}
	// init mean
	for (int i = 0; i < dim_ori; i++){
		mean[i] = random(10000);
	}

	// save data project mean dst

	gpu_init(0);

	double start = wallclock();

	gpu_pca_gzn(project, mean, data, dst, num_data, dim_pca, dim_ori);

	printf("Time = %lfs\n", wallclock() - start);

	// save data project mean dst
	char name[256] = "data/data.txt";
	char nameProject[256] = "data/project.txt";
	char nameMean[256] = "data/mean.txt";
	char nameDst[256] = "data/dst.txt";

	saveMat(data, dim_ori, num_data, name);
	saveMat(project, dim_pca, dim_ori, nameProject);
	saveMat(mean, dim_ori, 1, nameMean);
	saveMat(dst, dim_pca, num_data, nameDst);

}