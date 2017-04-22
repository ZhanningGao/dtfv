from liblinearutil import *

import glob
import numpy as np
import os.path
import os

UCF_SAVE_FILE_PATH = "/home/gzn/Desktop/data/zhanning/datasets/UCF-101"
UCF_FV_PATH   = "/home/gzn/Desktop/data/zhanning/UCF_FV"

UCF_TRTE_LIST = "/home/gzn/Desktop/data/zhanning/datasets/UCF-101/ucfTrainTestlist"

IDT_TYPE = ["traj","hog","hof","mbhx","mbhy"]
BACK_END = ".fv.txt"

def getClassInd(classIndFile):
    with open(classIndFile) as classInd:
        classNames = classInd.readlines()
    Class_Ind = {}
    for ind_className in classNames:
        val, key = ind_className.strip().split(' ')
        Class_Ind[key] = int(val)
    return Class_Ind

def getFVsGroup(testVids):
    Class_Ind = getClassInd(os.path.join(UCF_TRTE_LIST,'classInd.txt'))

# estimate dim of FV
    fv = []
    for IDTtype in IDT_TYPE:
        fvName = testVids[0].strip().split('/')[-1].split(' ')[0] + '.' + IDTtype + BACK_END
        fv.extend(np.loadtxt(os.path.join(UCF_FV_PATH, fvName)))

    dim_fv = len(fv)
    num_fv = len(testVids)

    testFVs = np.zeros((num_fv,dim_fv),dtype=np.float32)

    labels = []
    for testVid in testVids:
        className = testVid.strip().split('/')[-2]

        fv = []
        for IDTtype in IDT_TYPE:
            fvName = testVid.strip().split('/')[-1].split(' ')[0] + '.' + IDTtype + BACK_END
            fv.extend(np.loadtxt(os.path.join(UCF_FV_PATH, fvName),dtype=np.float32))
        testFVs[len(labels)] = np.array(fv)
        labels.append(Class_Ind[className])
        if len(labels)%1000 == 0:
            print '%d Videos done!'%(len(labels))

    print 'Read group done!'
    return testFVs, np.array(labels)

def getVidsGroup(ListFiles):
    with open(os.path.join(UCF_TRTE_LIST,ListFiles)) as list:
        testVidsGroup = (list.readlines())
    return testVidsGroup

def pre_process_data(testListFiles):

    testFVsGroup, testLabelsGroup = getFVsGroup(getVidsGroup(testListFiles))

    return testFVsGroup, testLabelsGroup

def readBin2Array(filename):

    print 'Reading bin files from %s'%(filename.split('/')[-1])

    DTYPE = filename.split('_')[-4]
    shape0 = int(filename.split('_')[-3])
    shape1 = int(filename.split('_')[-2])

    res = np.fromfile(filename,dtype=DTYPE)

    if shape1 == 0:
        res.shape = (shape0,)
    else:
        res.shape = (shape0,shape1)

    return res

if __name__ == '__main__':
    testListFiles = ['testlist01.txt','testlist02.txt','testlist03.txt']
    trainListFiles = ['trainlist01.txt','trainlist02.txt','trainlist03.txt']

    for i in range(len(testListFiles)):
        testListFile = testListFiles[i]
        trainListFile = trainListFiles[i]

        # check if saved X_te Y_te bin files
        if len(glob.glob(os.path.join(UCF_SAVE_FILE_PATH, 'Y_te_' + str(i) + '*'))) == 0:

            X_te, Y_te = pre_process_data(testListFile)

            X_te_name = os.path.join(UCF_SAVE_FILE_PATH, 'X_te_%d_%s_%d_%d_.bin' % (i, str(X_te.dtype), X_te.shape[0], X_te.shape[1]))
            Y_te_name = os.path.join(UCF_SAVE_FILE_PATH, 'Y_te_%d_%s_%d_%d_.bin' % (i, str(Y_te.dtype), Y_te.shape[0], 0))

        #save X_te Y_te
            X_te.tofile(X_te_name)
            Y_te.tofile(Y_te_name)
        else:
            X_te_name = glob.glob(os.path.join(UCF_SAVE_FILE_PATH, 'X_te_' + str(i) + '*'))[0]
            Y_te_name = glob.glob(os.path.join(UCF_SAVE_FILE_PATH, 'Y_te_' + str(i) + '*'))[0]

            X_te = readBin2Array(X_te_name)
            Y_te = readBin2Array(Y_te_name)

        # check if saved X_tr Y_tr bin files
        if len(glob.glob(os.path.join(UCF_SAVE_FILE_PATH, 'Y_tr_' + str(i) + '*'))) == 0:

            X_tr, Y_tr = pre_process_data(trainListFile)

            X_tr_name = os.path.join(UCF_SAVE_FILE_PATH, 'X_tr_%d_%s_%d_%d_.bin' % (i, str(X_tr.dtype), X_tr.shape[0], X_tr.shape[1]))
            Y_tr_name = os.path.join(UCF_SAVE_FILE_PATH, 'Y_tr_%d_%s_%d_%d_.bin' % (i, str(Y_tr.dtype), Y_tr.shape[0], 0))

        #save X_tr Y_tr
            X_tr.tofile(X_tr_name)
            Y_tr.tofile(Y_tr_name)
        else:
            X_tr_name = glob.glob(os.path.join(UCF_SAVE_FILE_PATH, 'X_tr_' + str(i) + '*'))[0]
            Y_tr_name = glob.glob(os.path.join(UCF_SAVE_FILE_PATH, 'Y_tr_' + str(i) + '*'))[0]

            X_tr = readBin2Array(X_tr_name)
            Y_tr = readBin2Array(Y_tr_name)



        m = train(problem(Y_tr, X_tr), parameter('-n 20 -c 1'))

        p_lable, p_acc, p_val = predict(Y_te, X_te, m)