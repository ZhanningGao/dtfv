import os
import os.path

UCF_path = "/home/gzn/Desktop/data/datasets/UCF101-resize"

UCF_list_path = "/home/gzn/Desktop/data/zhanning/datasets/UCF-101"

f = open(os.path.join(UCF_list_path,"UCFList_resize.txt"),'w')

for parent,dirnames,filenames in os.walk(UCF_path):
    for filename in filenames:
        f.write(os.path.join(parent,filename)+"\n")

f.close()