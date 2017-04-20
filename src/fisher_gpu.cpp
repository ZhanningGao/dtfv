#include "fisher_gpu.h"

using namespace std;

#define TYPE float

/////////////////////

FisherVector::FisherVector(string pcaDictFile, string codeBookFile)   {
    //load codeBook
    ifstream fin;
    fin.open(codeBookFile.c_str());
    if (!fin.is_open()) {
        cout<<"Cannot load "<<codeBookFile<<endl;
        exit(-1);
    }
    fin>>dimension>>numClusters;
    means = new TYPE[numClusters * dimension];
    covariances = new TYPE[numClusters * dimension];
    priors = new TYPE[numClusters];

    for (int i = 0; i < numClusters * dimension; i++)
        fin >> means[i];
    for (int i = 0; i < numClusters * dimension; i++)
        fin >> covariances[i];
    for (int i = 0; i < numClusters; i++)
        fin >> priors[i];
    fin.close();

    //load pca
    fin.open(pcaDictFile.c_str());
    if (!fin.is_open()) {
        cout<<"Cannot open "<<pcaDictFile<<endl;
        exit(-1);
    }
    fin>>dim_ori>>dim_pca;
    pcaMat = new TYPE[dim_ori * dim_pca];
    pcaMatmean = new TYPE[dim_ori];
 //   for (int i = 0; i < dim_ori*dim_pca; i++)
 //       fin>>pcaMat[i];
    for (int r = 0; r < dim_pca; r++){
        for (int c = 0; c < dim_ori; c++)
        {
            fin>>pcaMat[c*dim_pca + r];
        }
    }
    for (int i = 0; i < dim_ori; i++)
        fin>>pcaMatmean[i];
    fin.close();

    if (dim_pca!=dimension)    {
        cout<<"PCA and GMM not match."<<endl;
        exit(-1);
    }

    numData = 0;

    initFV();

}

FisherVector::~FisherVector()   {
    if (pcaMat)
        delete [] pcaMat;
    if (pcaMatmean)
        delete [] pcaMatmean;
    if (means)
        delete [] means;
    if (means)
        delete [] priors;
    if (means)
        delete [] covariances;
}

bool FisherVector::initFV()  {
    // allocate space
    int dimPerPyr = dimension * numClusters * 2;        // 2KD

    if (fv.size() > 0)
        fv.clear();
    fv.resize(dimPerPyr, 0.0);
    return true;
}

bool FisherVector::encodeFV(TYPE *data, vl_size numData){
    this->numData = numData;

    // pca
    TYPE *dataPca = new TYPE[numData*dim_pca];

    gpu_pca_gzn(pcaMat, pcaMatmean, data, dataPca, numData, dim_pca, dim_ori);

    int res = 0;
    TYPE *enc = new float[2*dimension*numClusters];

    res = my_fisher_encode_gpu(enc, means, dimension, numClusters, covariances, priors, numData, dataPca);

    for (int dim = 0; dim < numClusters*dimension*2; dim++)
        fv[dim] = enc[dim];

    delete [] enc;
    delete [] dataPca;

}

void FisherVector::my_get_gmm_data_posteriors_f_gpu(TYPE * posteriors,
				 vl_size numClusters,
				 vl_size numData,
				 TYPE const * priors,
				 TYPE const * means,
				 vl_size dimension,
				 TYPE const * covariances,
				 TYPE * enc,
				 TYPE* sqrtInvSigma, TYPE* data)
{
    TYPE halfDimLog2Pi = (dimension / 2.0) * log(2.0*VL_PI);
    // TYPE * logCovariances ;
    // TYPE * logWeights ;
    // TYPE * invCovariances ;
    TYPE * posteriors_g; // = (TYPE*) vl_malloc(sizeof(TYPE)* numClusters * numData);

    gpu_gmm_1( covariances,  priors, means, posteriors_g, numClusters, dimension, numData, halfDimLog2Pi, enc, sqrtInvSigma, data) ;
}

int FisherVector::my_fisher_encode_gpu (	
			  TYPE * enc_g,
			  TYPE const * means, vl_size dimension, vl_size numClusters,
			  TYPE const * covariances,
			  TYPE const * priors,
			  vl_size numData, TYPE* data) {

    vl_size numTerms = 0 ;
    TYPE * posteriors ;
    TYPE * sqrtInvSigma;

    assert(numClusters >= 1) ;
    assert(dimension >= 1) ;

    gpu_init(0);

    gpu_copy(covariances, priors, means, numClusters, dimension) ;

    my_get_gmm_data_posteriors_f_gpu(posteriors, numClusters, numData,
				   priors,
				   means, dimension,
				   covariances,
				   enc_g, sqrtInvSigma, data) ;
    gpu_free();

    return numTerms ;
}

TYPE *FisherVector::loadData(string dataFile, vl_size &numData, vl_size &dimension)    {
    ifstream fin;
    fin.open(dataFile.c_str());
    if (!fin.is_open()) {
        cout<<"Cannot open "<<dataFile<<endl;
        return NULL;
    }

    vector<vector<TYPE> > inputData;
    string line;
    stringstream ss;
    TYPE val;
    while (getline(fin, line))  {
        ss<<line;
        vector<TYPE> feat;
        while (ss>>val)
            feat.push_back(val);
        inputData.push_back(feat);
        ss.clear();
        ss.str("");
    }
    fin.close();

    numData = inputData.size();
    dimension = inputData[0].size();

    TYPE *data = new TYPE[numData*dimension];
    for (int dataIdx = 0; dataIdx < numData; dataIdx++) {
        for (int d = 0; d < dimension; d++) {
            data[dataIdx*dimension+d] = inputData[dataIdx][d];
        }
    }
    return data;
}

TYPE *FisherVector::loadDataIDT(string dataFile, vl_size &numData, vl_size &dimension)    {
// unfinished
    TYPE *data = new TYPE[numData*dimension];
    for (int dataIdx = 0; dataIdx < numData; dataIdx++) {
        for (int d = 0; d < dimension; d++) {
            data[dataIdx*dimension+d] = 0.0;
        }
    }
    return data;
}

bool FisherVector::clearFV(){
    numClusters = 0;
    dim_pca = 0;
    dim_ori = 0;
    fv.clear();
    numData = 0;
    return true;
}

vector<TYPE> &FisherVector::getFV()   {
    if (this->numData < 1)
        return fv;
    // normalize by sqrt and l2 for each component separately
        int ptr = 0;
        TYPE fv1_sum = 0.0;
        TYPE fv2_sum = 0.0;
        for (int cluster = 0; cluster < numClusters; cluster++)   {
            TYPE prefix1 = 1/(numData*sqrt(priors[cluster]));
            TYPE prefix2= 1/(numData*sqrt(2.0*priors[cluster]));
            for (int dim = 0; dim < dimension; dim++) {
                // fill the missing fv terms
                fv[ptr+cluster*dimension+dim] *= prefix1;
                fv[ptr+dimension*numClusters + cluster*dimension+dim] *= prefix2;
                // sqrt norm
                if (fv[ptr+cluster*dimension+dim] > 0)
                    fv[ptr+cluster*dimension+dim] = sqrt(fv[ptr+cluster*dimension+dim]);
                else
                    fv[ptr+cluster*dimension+dim] = -sqrt(-fv[ptr+cluster*dimension+dim]);

                if (fv[ptr+dimension*numClusters + cluster*dimension+dim] > 0)
                    fv[ptr+dimension*numClusters + cluster*dimension+dim] = sqrt(fv[ptr+dimension*numClusters + cluster*dimension+dim]);
                else
                    fv[ptr+dimension*numClusters + cluster*dimension+dim] = -sqrt(-fv[ptr+dimension*numClusters + cluster*dimension+dim]);
            }
        }
        // l2 norm
        for (int dim = 0; dim < dimension * numClusters; dim++) {
            fv1_sum += fv[ptr+dim] * fv[ptr+dim];
            fv2_sum += fv[ptr+dimension * numClusters+dim] * fv[ptr+dimension * numClusters+dim];
        }
        fv1_sum = sqrt(fv1_sum);
        fv2_sum = sqrt(fv2_sum);
        if (fv1_sum > 1e-10 && fv2_sum > 1e-10) {
            for (int dim = 0; dim < dimension * numClusters; dim++) {
                fv[ptr+dim] /= fv1_sum;
                fv[ptr+dimension * numClusters+dim] /= fv2_sum;
            }
        }
    return fv;
}


/*template <class TT>
void copyVec2Arr(vector<TT> fvec, TT* arr){
  typename vector<TT>::iterator iter = fvec.begin();
  int index = 0;
  for(iter;iter != fvec.end(); iter++)
    {
      arr[index++] = *iter;
    }
}


int VL_CAT(fun, f)(int a, int b)
{
  return a+b;
}*/
