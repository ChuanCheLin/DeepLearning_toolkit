#Author: Eric Lin
#made for few-shot learning dataset arrangement
#last edit time: 2022/03/22
#create specific JPEGImages folder

def duplicate(all_jpg_path = "/home/eric/mmdetection/data/VOCdevkit/datasets/VOC2007/JPEGImages/", output_jpg_path = str, ann_path = str):
    from tqdm import tqdm
    import os
    dirs = os.listdir(ann_path)
    lines = (line.rstrip('.xml') for line in dirs)

    count=0
    #make JPEGImages folder by Annotations given to use voc2coco.py
    import shutil
    pathnew = output_jpg_path  #need change
    if(os.path.isdir(pathnew)==False):
        os.mkdir(pathnew)

    for line in tqdm(lines):
        source= all_jpg_path + '/' + line + '.jpg'
        destination = pathnew + '/' + line + '.jpg'
        shutil.copy(source, destination)
        count= count+1

    print('finish duplicating ' + str(count) + " imgs to " + pathnew)