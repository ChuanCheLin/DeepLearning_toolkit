# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:31:47 2020

@author: Chun
"""
import csv
import os
import numpy as np
from label_operator import label_operator as lab

from pascal_voc_io import PascalVocReader as PR
from pascal_voc_io import PascalVocWriter as PW


def change(filepath, box, label_old, label_new, new_dir='data0504/Annotations'):
    print(filepath)
    pr = PR(filepath)
    shapes = pr.getShapes()
    sl = pr.getSources()
    
    pw = PW(sl['folder'],
            sl['filename'],
            sl['size'],
            databaseSrc='Unknown',
            localImgPath=sl['path'])
    pw.verified = True
    
    for shape in shapes:
        if shape[0] == label_old:
            sourbox = list(shape[1][0])
            sourbox.extend(shape[1][2])
            #print(np.array(box), np.array(sourbox))
            if np.array_equal(np.array(box), np.array(sourbox)):
                label = label_new
            else:
                label = shape[0]
        else:
            label = shape[0]
            
        difficult = 0
        bndbox = shape[1]
        pw.addBndBox(bndbox[0][0], bndbox[0][1], bndbox[2][0], bndbox[2][1], label, difficult)
    
    filename = os.path.basename(filepath)
    newfile = os.path.join(new_dir, filename)
    # D:\Chun\data\data0504\new_anno
    pw.save(newfile)

def read_table(csvf):
    dics = []
    i = 0
    with open(csvf, newline='') as f:
        
        rows = csv.reader(f)
            
        for row in rows:
            if i == 0:
                i+=1
                continue
            dic = {}        
            dic['image'] = row[0]
            dic['ABC'] = row[1]
            dic['severe'] = row[2]
            dic['change'] = row[3]
            dic['box'] = list(map(int,row[:-1][4:]))
            dic['memo'] = row[-1]
            
            dics.append(dic)
        
    return dics

csvf = 'data0504\Rae0502.csv'

i = 0
dics = read_table(csvf)

ABC = "ABCDEFGHIJKL"

labels = {
           '1':'mosquito_early',
           '2':'mosquito_late',
           '1.早期':'mosquito_early',
           '2.晚期':'mosquito_late',
           }

xml_dir = 'data0504\Annotations'

for d in dics:
    
    if d['change'] is not '':
        
        xml = d['image'][:-4] + '.xml'
        print('changing file', xml)
        
        xml_path = os.path.join(xml_dir, xml)
        change(xml_path, d['box'],
               labels[d['severe']], labels[d['change']])
        print('changing label from %s to %s' 
            %(labels[d['severe']], labels[d['change']]))




#   -------------------------------------------
coco = {
        "info":{},
        "license":[],
        
        "images":[
                   {
                    "file_name":"0005.jpg",
                    "height":   3072,
                    "width":    4096,
                    "id":0,
                    },
                   {
                    "file_name":"0007.jpg",
                    "height":   3072,
                    "width":    4096,
                    "id":1,
                    }, 
            ],
        
        "annotations":[
                    {
                    "segmentation":[],
                    "area":    60000,
                    "image_id":0,
                    "bbox": [100,100,300,400],
                    "category_id": 1,
                    "id":0,
                    },
                   {
                    "segmentation":[],
                    "area":    60000,
                    "image_id":0,
                    "bbox": [200,300,400,600],
                    "category_id": 1,
                    "id":1,
                    }, 
                   {
                    "segmentation":[],
                    "area":    60000,
                    "image_id":1,
                    "bbox": [100,100,300,400],
                    "category_id": 1,
                    "id":2,
                    },
                   {
                    "segmentation":[],
                    "area":    60000,
                    "image_id":1,
                    "bbox": [200,300,400,600],
                    "category_id": 1,
                    "id":3,
                    },
            ],
        
        "categories":[
                    {       
                    "supercategory": 'leaf',
                    "id": 1,
                    "name": "brownblight",
                    },
                    {       
                    "supercategory": 'leaf',
                    "id": 2,
                    "name": "blister",
                    },
            ],
        }
print("ddd")
        
        
        
        
        
        
        