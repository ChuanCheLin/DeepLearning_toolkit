#Author: Eric Lin
#made for few-shot learning dataset arrangement
#last edit time: 2021/10/08
#to check the dataset distribution is correct
import os
def Diff(li1, li2): 
    return (list(set(li1).symmetric_difference(set(li2))))
def Same(li1, li2): 
    return (list(set(li1).intersection(set(li2))))

    
path1 = "/home/eric/mmdetection/data/VOCdevkit/datasets/VOC2007/JPEGImages/"
dirs1 = os.listdir(path1)
lines1 = (line.rstrip('.jpg') for line in dirs1)

path2 ="/home/eric/mmdetection/data/VOCdevkit/datasets/VOC2007/Annotations/"
dirs2 = os.listdir(path2)
lines2 = (line.rstrip('.xml') for line in dirs2)


diff = Diff(lines1, lines2)
print(diff)
# same = Same(lines1, lines2)
# print(same)
#can't be called in the same time(don't know the reason)
    


    