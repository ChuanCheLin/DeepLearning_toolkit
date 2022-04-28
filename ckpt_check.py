# design for checking the details of ckpt files
import torch
import argparse
import os

ckpt_path1 = "/home/eric/few-shot-object-detection/checkpoints/coco/faster_rcnn/set1/split2/60shot_2stage_test/combined/model_reset_combine.pth"
ckpt_path2 = "/home/eric/few-shot-object-detection/checkpoints/coco/faster_rcnn/set1/split2/base/model_final.pth"
ckpt1 = torch.load(ckpt_path1)
ckpt2 = torch.load(ckpt_path2)
print(ckpt1.keys())
print(ckpt2.keys())

list1 = []

for name in ckpt1['model']:
    list1.append((ckpt1['model'][name]))

list2 = []

for name in ckpt2['model']:
    list2.append((ckpt2['model'][name]))

count = 0 
for com in list1:
    if torch.equal(com.to(torch.device("cpu")), list2[count].to(torch.device("cpu"))) == False:
        print(len(list1[count]))
        print(len(list2[count]))
    count +=1