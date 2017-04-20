import os.path
import glob

TVID_path_test_tr = "/home/gzn/Desktop/data/datasets/TRECVID14/DR-TS/test_tr"
TVID_path_train_tr = "/home/gzn/Desktop/data/datasets/TRECVID14/PS-TR/training/ps_tr"
TVID_path_bg = "/home/gzn/Desktop/data/datasets/TRECVID14/PS-TR/training/ps_tr/event_tr/bg"

TVID_path_test_tr_resize = "/home/gzn/Desktop/data/datasets/TRECVID14/DR-TS_resize/test_tr"
TVID_path_train_tr_resize = "/home/gzn/Desktop/data/datasets/TRECVID14/PS-TR_resize/training/ps_tr"
TVID_path_bg_resize = "/home/gzn/Desktop/data/datasets/TRECVID14/PS-TR_resize/training/ps_tr/event_tr/bg"

TVID_list_path = "/home/gzn/Desktop/data/zhanning/datasets/TRECVID14"

if not os.path.exists(TVID_list_path):
    os.mkdir(TVID_list_path)

f = open(os.path.join(TVID_list_path,"TVIDList.txt"),'w')
f_resize = open(os.path.join(TVID_list_path,"TVIDList_resize.txt"),'w')

for filename in glob.glob(os.path.join(TVID_path_test_tr,"*.avi")):
    f.write(filename + "\n")
    f_resize.write(os.path.join(TVID_path_test_tr_resize,os.path.basename(filename)) + "\n")

for filename in glob.glob(os.path.join(TVID_path_train_tr,"*.avi")):
    f.write(filename + "\n")
    f_resize.write(os.path.join(TVID_path_train_tr_resize,os.path.basename(filename)) + "\n")

for filename in glob.glob(os.path.join(TVID_path_bg, "*.avi")):
    f.write(filename + "\n")
    f_resize.write(os.path.join(TVID_path_bg_resize,os.path.basename(filename)) + "\n")

f.close()
f_resize.close()