from hashlib import new
from importlib.resources import path
import xml.etree.ElementTree as ET
import os
ann_path = "/home/eric/mmdetection/data/VOCdevkit/datasets/VOC2007/annotation_caloptilia/"

for filename in os.listdir(ann_path):
    xml_name = ann_path + filename
    tree = ET.parse(xml_name)
    root = tree.getroot()
    # print(root[1].text)
    oldname1 = root[1].text
    # print(root[2].text)
    oldname2 = root[2].text

    newname1 = oldname1.replace("茶細蛾",'')
    newname2 = oldname2.replace("茶細蛾",'')

    root[1].text = newname1
    root[2].text = newname2
    tree.write(xml_name)