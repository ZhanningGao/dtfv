#ifndef COMPUTE_FV_GPU_H
#define COMPUTE_FV_GPU_H

#define TYPE float

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
#include "pca.h"
#include <random>
//#include "sift.h"
//include "mydsift.h"
//#include "CommonTools.h"
#include "omp.h"

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


using namespace std;

class FisherVector  {
    public:
        FisherVector(string pcaDictFile, string codeBookFile);
        ~FisherVector();
        bool initFV();
        bool encodeFV(TYPE * data, vl_size numData);
        vector<TYPE> & getFV();
        bool clearFV();
		void my_get_gmm_data_posteriors_f_gpu(TYPE * posteriors,
				 vl_size numClusters,
				 vl_size numData,
				 TYPE const * priors,
				 TYPE const * means,
				 vl_size dimension,
				 TYPE const * covariances,
				 TYPE * enc,
				 TYPE* sqrtInvSigma, TYPE* data);
		int my_fisher_encode_gpu (	
			  TYPE * enc_g,
			  TYPE const * means, vl_size dimension, vl_size numClusters,
			  TYPE const * covariances,
			  TYPE const * priors,
			  vl_size numData, TYPE* data);

		TYPE *loadData(string dataFile, vl_size &numData, vl_size &dimension);
		TYPE *loadDataIDT(string dataFile, vl_size &numData, vl_size &dimension);

        TYPE *getPcaMat(){
            return pcaMat;
        }
        TYPE *getPcaMean(){
            return pcaMatmean;
        }
        TYPE *getClusters(){
            return means;
        }
        TYPE *getPriors(){
            return priors;
        }
        TYPE *getCovariances(){
            return covariances;
        }

        int getPcaDim(){
            return dim_pca;
        }
        int getNumClusters(){
            return numClusters;
        }
        int getOriDim(){
            return dim_ori;
        }
        int getNumData(){
            return numData;
        }

    private:
    	int numClusters;
    	int dimension;
    	int dim_pca;
    	int dim_ori;

    	TYPE *means;
    	TYPE *covariances;
    	TYPE *priors;

    	TYPE *pcaMat;
    	TYPE *pcaMatmean;

        vector<TYPE> fv;
        int numData;
};

#endif
