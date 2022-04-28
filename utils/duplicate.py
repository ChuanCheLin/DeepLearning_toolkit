#Author: Eric Lin
#made for few-shot learning dataset arrangement
#last edit time: 2022/03/22
#create specific JPEGImages folder

from tqdm import tqdm
import os
#Image dir to filt from
path1 = "/home/eric/mmdetection/data/VOCdevkit/datasets/VOC2007/JPEGImages/" #All imgs
#Annotation dir to filt 
path2 = "/home/eric/mmdetection/data/VOCdevkit/datasets/VOC2007/Annotations_split1_base/"  #need change
dirs2 = os.listdir(path2)
lines2 = (line.rstrip('.xml') for line in dirs2)



count=0
#make JPEGImages folder by Annotations given to use voc2coco.py
import shutil
pathnew = "/home/eric/mmdetection/data/VOCdevkit/datasets/VOC2007/JPEGImages_split1_base/"  #need change
if(os.path.isdir(pathnew)==False):
    os.mkdir(pathnew)

for line in tqdm(lines2):
    source= path1 + '/' + line + '.jpg'
    destination = pathnew + '/' + line + '.jpg'
    shutil.copy(source, destination)
    count= count+1

print('finish duplicating ' + str(count) + " imgs to " + pathnew)