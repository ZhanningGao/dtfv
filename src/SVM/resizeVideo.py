# extract DT features on the fly and select feature points randomly
# need select_pts and dt binaries

import subprocess, os, ffmpeg
import sys
import time
import shutil

# Dense Trajectories binary
dtBin = '../IDT/release/DenseTrackStab'
# Compiled fisher vector binary
fvBin = '../compute_fv_gpu'
# Temp directory to store resized videos
tmpDir = '/home/gzn/Desktop/data/datasets/UCF101-resize'
# Process ID for running in parallel
pID = 0
# PCA list
pcaList = '../../data/ucfSVM.pca.lst'
# GMM list
codeBookList = '../../data/ucfSVM.codebook.lst'

def extract(videoName):
    if not os.path.exists(videoName):
        print '%s does not exist!' % videoName
        return False
    pardir = videoName.split('/')
    pardir = pardir[-2]
    if not os.path.exists(os.path.join(tmpDir, pardir)):
        os.mkdir(os.path.join(tmpDir, pardir))
    resizedName = os.path.join(tmpDir, pardir, os.path.basename(videoName))
    if not ffmpeg.resize(videoName, resizedName):
        shutil.copy(videoName, resizedName)
    return True

if __name__ == '__main__':
    videoList = "/home/gzn/Desktop/data/zhanning/datasets/UCF-101/UCFList.txt"
    try:
        f = open(videoList, 'r')
        videos = f.readlines()
        f.close()
        videos = [video.rstrip() for video in videos]
        for i in range(0, len(videos)):
            print pID, videos[i]
            tic = time.time()
            extract(videos[i])
            print "Running time: %fs" % (time.time()-tic)
    except IOError:
        sys.exit(0)
