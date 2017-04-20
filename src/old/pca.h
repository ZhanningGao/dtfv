#ifndef PrinCompAna_H
#define PrinCompAna_H

#define TYPE float

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>

#include "alglib/dataanalysis.h"

using namespace std;

class PCAWrapper    {
    public:
        PCAWrapper(bool whitening = false);
        PCAWrapper(string fileName, bool whitening = false);
        ~PCAWrapper();
        
        vector<TYPE> project(vector<TYPE> input);
        bool train(vector<vector<TYPE> > &inputData, int pDim, string outputFile);
        int getDim();
    private:
        TYPE *projMat;
        TYPE *aveVec;
        TYPE *eigVec; // inverse of eigen values's square roots
        size_t pDim;
        size_t oDim;
        bool whitening;
};

#endif
