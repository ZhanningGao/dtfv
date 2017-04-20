#ifndef GaussianMM_H
#define GaussianMM_H

#define TYPE float

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>

extern "C"  {
#include "vl/gmm.h"
}

using namespace std;

class GMMWrapper    {
    public:
        GMMWrapper();
        GMMWrapper(string codeBookName);
        ~GMMWrapper();

        bool train(string dataFile, vl_size numClusters, string codeBookName);
    private:
        TYPE *loadData(string dataFile, vl_size &numData, vl_size &dimension);
    public:
        TYPE *means;
        TYPE *covs;
        TYPE *priors;
        int dimension;
        int numClusters;
};

#endif
