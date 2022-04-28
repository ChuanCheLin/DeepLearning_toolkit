# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:03:13 2020

@author: Chun
"""


import cv2 
import numpy as np
import torch
import torchvision
from torchvision import transforms as tf
from PIL import Image, ImageOps
from PyQt5.QtGui import QImage


im = 'images/5603.jpg'

tran = tf.Compose([tf.ToTensor(),])

imq = QImage()
imq.load(im)
icolor = imq.pixelColor(0,0)
icolor = np.array([icolor.red(), icolor.green(), icolor.blue() ])
print(icolor)
impil = Image.open(im)
pill = lambda x,y : np.array(impil.getpixel((x, y)))
print(pill(0,0))
impil2 = ImageOps.exif_transpose(impil)
pill2 = lambda x,y : np.array(impil2.getpixel((x, y)))
print(pill2(0,0))

imcv = cv2.imread(im)
print(imcv[0,0])
imcv2 = cv2.imread(im, cv2.IMREAD_UNCHANGED)
print(imcv2[0,0])