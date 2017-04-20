#include <ctime>

#include "fisher_gpu.h"
#include "feature_f.h"

#include "utils.h"

/*
 * A sample use of fisher vector coding with DT.
 * Require 5 PCA projection matrices and 5 GMM codebooks 
 * for each component of DT: TRAJ, HOG, HOF, MBHX, MBHY
 * Read input from stdin
 */

using namespace std;

#define MAX_NUM_DATA 100000

int main(int argc, char **argv) {
    if (argc < 4)   {
        cout<<"Usage: "<<argv[0]<<" pcaList codeBookList outputBase"<<endl;
        return 0;
    }
    // Important: GMM uses OpenMP to speed up
    // This will cause problem on cluster where all
    // cores from a node run this binary
    vl_set_num_threads(1);
    string pcaList(argv[1]);
    string codeBookList(argv[2]);
    string outputBase(argv[3]);
    string types[5] = {"traj", "hog", "hof", "mbhx", "mbhy"};
    vector<FisherVector*> fvs(5, NULL);

    //FV para
    int numClusters;
    vector<int> dimension;
    //time
    clock_t t1,t2, t3;

    ifstream fin1, fin2;
    fin1.open(pcaList.c_str());
    if (!fin1.is_open())    {
        cout<<"Cannot open "<<pcaList<<endl;
        return 0;
    }
    fin2.open(codeBookList.c_str());
    if (!fin2.is_open())    {
        cout<<"Cannot open "<<codeBookList<<endl;
        return 0;
    }
    string pcaFile, codeBookFile;
    for (int i = 0; i < fvs.size(); i++)    {
        getline(fin1, pcaFile);
        getline(fin2, codeBookFile);
        fvs[i] = new FisherVector(pcaFile, codeBookFile);
        fvs[i]->initFV();  // 1 layer of spatial pyramids

      //  printf("%dth numClusters = %d\n", i+1, fvs[i]->getNumClusters());
      //  printf("%dth OriDim = %d\n", i+1, fvs[i]->getOriDim());
      //  printf("%dth PcaDim = %d\n", i+1, fvs[i]->getPcaDim());
        
        //saveMatUtils(fvs[i]->getPcaMat(), fvs[i]->getOriDim(), fvs[i]->getPcaDim(), "pca_debug.txt");
    }
    fin1.close();
    fin2.close();

    t1 = clock();

    string line;
    //load data
    float *Traj, *Hog, *Hof, *Mbhx, *Mbhy;
    Traj = new float[TRAJ_DIM *MAX_NUM_DATA];
    Hog  = new float[HOG_DIM  *MAX_NUM_DATA];
    Hof  = new float[HOF_DIM  *MAX_NUM_DATA];
    Mbhx = new float[MBHX_DIM *MAX_NUM_DATA];
    Mbhy = new float[MBHY_DIM *MAX_NUM_DATA];

    float *pTraj, *pHog, *pHof, *pMbhx, *pMbhy;
    pTraj = Traj;
    pHog  = Hog;
    pHof  = Hof;
    pMbhx = Mbhx;
    pMbhy = Mbhy;

    int numData = 0;

    while (getline(cin, line))  {
        DTFeature_f feat(line);
        //TODO: Store feature of DT
        memcpy(pTraj, feat.traj, sizeof(float)*TRAJ_DIM); pTraj += TRAJ_DIM;
        memcpy(pHog,  feat.hog, sizeof(float)*HOG_DIM); pHog += HOG_DIM;
        memcpy(pHof,  feat.hof, sizeof(float)*HOF_DIM); pHof += HOF_DIM;
        memcpy(pMbhx, feat.mbhx, sizeof(float)*MBHX_DIM); pMbhx += MBHX_DIM;
        memcpy(pMbhy, feat.mbhy, sizeof(float)*MBHY_DIM); pMbhy += MBHY_DIM;


        numData++;

        if (numData == MAX_NUM_DATA) {
            printf("%s\n", "the number of IDT points have reached the MAX_NUM_DATA");
            break;
        }
    }

    t3 = clock();

    fvs[0]->encodeFV(Traj, numData);
    fvs[1]->encodeFV(Hog, numData);
    fvs[2]->encodeFV(Hof, numData);
    fvs[3]->encodeFV(Mbhx, numData);
    fvs[4]->encodeFV(Mbhy, numData);
//    for (int i=0; i<TRAJ_DIM; i++){
//        printf("%0.4f\t", Traj[i]);
//    }
    printf("numData = %d\n", fvs[4]->getNumData());

    delete [] Traj;
    delete [] Hog;
    delete [] Hof;
    delete [] Mbhx;
    delete [] Mbhy;
    
    t2 = clock();
    cout<< "Runing time for loading data: " << (float)(t3-t1)/CLOCKS_PER_SEC << "s" <<endl;
    cout<< "Runing time for FV: " << (float)(t2-t3)/CLOCKS_PER_SEC << "s" <<endl;

    cout<< "outputBase: " << outputBase << endl;
    
    cout<<"Points load complete."<<endl;
    for (int i = 0; i < fvs.size(); i++)    {
        ofstream fout;
        string outName = outputBase + "." + types[i] + ".fv.txt";
        fout.open(outName.c_str());
        vector<float> fv = fvs[i]->getFV();
        fout<<fv[0];
        for (int j = 1; j < fv.size(); j++)
            fout<<" "<<fv[j];
        fout<<endl;
        fout.close();
        fvs[i]->clearFV();
    }

    for (int i = 0; i < fvs.size(); i++)
        delete fvs[i];
    return 0;
}
