#Author: Eric Lin
import os
    
path = "/home/eric/mmdetection/data/VOCdevkit/datasets/VOC2007/JPEGImages/"
dirs = os.listdir(path)
for name in dirs:
    if '.JPG' in name:
        mid = name.rstrip('.JPG')
        oldname = path + '/' + mid + '.JPG'
        newname = path + '/' + mid + '.jpg'
        os.rename(oldname, newname)
