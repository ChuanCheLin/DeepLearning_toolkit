#Author: Eric Lin
#made for dataset arrangement
#last edit time: 2021/10/08
import os
    
path = "/home/eric/mmdetection/data/VOCdevkit/datasets/VOC2007/JPEGImages/"
dirs = os.listdir(path)
for name in dirs:
    if '(' in name:
        old = name
        new = name.replace('(','')
        new = new.replace(')','')
        new = new.replace(' ','')
        oldname = path + '/' + old 
        newname = path + '/' + new 
        #print(oldname)
        #print(newname)
        os.rename(oldname, newname)