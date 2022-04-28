# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:28:15 2020

@author: Chun
"""
import os
from PIL import Image
import matplotlib.pyplot as plt

im_list = os.listdir('./tempan')
exist = os.listdir('./data0504/out')
exist.append('0849.jpg')
exist.append('0950.jpg')
exist.append('0884.jpg')
exist.append('1195.jpg')
exist.append('1215.jpg')
exist.append('1251.jpg')
exist.append('1280.jpg')
for im in im_list:
    
    if im in exist:
        continue
    
    
    plt.figure(figsize=(12,9))
    
    plt.subplot(221)
    img1 = Image.open('./tempan/'+im)
    plt.imshow(img1)
    plt.axis('off') 
    plt.title('ground truth',y=-0.08 )
    
    plt.subplot(222)
    try:
        img2 = Image.open('./data0504/FRCNN/tmp/'+im)
    except:
        plt.close()
        continue
    plt.imshow(img2)
    plt.axis('off') 
    plt.title('faster R-CNN',y=-0.08 )
    
    plt.subplot(223)
    try:
        img3 = Image.open('./data0504/CRCNN/tmp/'+im)
    except:
        plt.close()
        continue
    plt.imshow(img3)
    plt.axis('off') 
    plt.title('Cascade R-CNN',y=-0.08 )
    
    plt.subplot(224)
    try:
        img4 = Image.open('./data0504/KLL/tmp/'+im)
    except:
        plt.close()
        continue
    plt.imshow(img4)
    plt.axis('off') 
    plt.title('F RCNN.K-L loss',y=-0.08 )
    
    plt.tight_layout()
    # plt.show()
    
    plt.savefig('./data0504/out/'+im,dpi=100)
    plt.close()
    print('process', im)