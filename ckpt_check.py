# design for checking the details of ckpt files
import torch
import argparse
import os

ckpt_path1 = "/home/eric/model_reset_surgery.pth"
ckpt_path2 = "/home/eric/model_final.pth"
ckpt1 = torch.load(ckpt_path1)
ckpt2 = torch.load(ckpt_path2)
print(ckpt1.keys())
print(ckpt2.keys())

list1 = []
nameset1 = []

for name in ckpt1['model']:
    list1.append((ckpt1['model'][name]))
    nameset1.append(name)
    print(name)

list2 = []
nameset2 = []

for name in ckpt2['model']:
    list2.append((ckpt2['model'][name]))
    nameset2.append(name)
    #print(name)

# check model parameters
# count = 0 
# for com in list1:
#     if torch.equal(com.to(torch.device("cpu")), list2[count].to(torch.device("cpu"))) == False:
#         print(len(list1[count]))
#         print(len(list2[count]))
#     count +=1

# check model names
# print(set(nameset1)-(set(nameset2)))
# print(set(nameset2)-(set(nameset1)))