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

def extract(videoName, resizedName):
    if not os.path.exists(videoName):
        print '%s does not exist!' % videoName
        return False
    if not os.path.exists(os.path.split(resizedName)[0]):
        os.makedirs(os.path.split(resizedName)[0])

    if not ffmpeg.resize(videoName, resizedName):
        shutil.copy(videoName, resizedName)
    return True

if __name__ == '__main__':
    videoList = "/home/gzn/Desktop/data/zhanning/datasets/TRECVID14/TVIDList.txt"
    resizeList = "/home/gzn/Desktop/data/zhanning/datasets/TRECVID14/TVIDList_resize.txt"
    try:
        f = open(videoList, 'r')
        fr = open(resizeList, 'r')
        videos = f.readlines()
        videos_resize = fr.readlines()
        f.close()
        fr.close()

        videos = [video.rstrip() for video in videos]
        videos_resize = [video.rstrip() for video in videos_resize]
        for i in range(0, len(videos)):
            print videos[i]
            tic = time.time()
            extract(videos[i],videos_resize[i])
            print "Running time: %fs" % (time.time()-tic)
    except IOError:
        sys.exit(0)
