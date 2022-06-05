# create csv file for a entire folder(might contain multiple training logs)
from pathlib import Path
import re
import numpy as np 
data_root = "/home/eric/few-shot-object-detection/checkpoints/coco/faster_rcnn/set1/split1/"

pathlist = Path(data_root).glob('**/*.txt')

all = []
length = int
names = []
for path in pathlist:
    name = str(path).replace(data_root,"").replace("/log.txt","")
    names.append(name)
    with open(path, 'r') as fp:
        data = fp.readlines()
        nums = []
        for i in range(len(data)):
            if "Per-category bbox AP:" in data[i]:
                id = i
        for j in range(6):
            for num in ([float(s) for s in re.findall(r'-?\d+\.?\d*', (data[id+j+3]))]):
                nums.append(num)
    length = len(nums)
    all.append(nums)

all_trans = []
all_trans.append(names)

print(all_trans)

for i in range(length):
    temp = []
    for num in all:
        temp.append(num[i])
    all_trans.append(temp)

np.savetxt("results.csv", all_trans, delimiter =",",fmt ='% s')