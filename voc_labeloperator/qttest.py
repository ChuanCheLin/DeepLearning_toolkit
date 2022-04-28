# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:14:48 2020

@author: Chun
"""


from PyQt5.QtGui import QImage
from PIL import Image, ImageOps
import numpy as np
import cv2
import os 


def check_rotate(imagepath):
    image = QImage()
    image.load(imagePath)
    
    impil = Image.open(imagePath)
    
    impil = ImageOps.exif_transpose(impil)
    pill = lambda x,y : np.array(impil.getpixel((x, y)))
    # print(pill(0,0))
    
    icolor = image.pixelColor(0,0)
    icolor = np.array([icolor.red(), icolor.green(), icolor.blue() ])
    imageShape = [image.height(), image.width(),
                  1 if image.isGrayscale() else 3]
    # print(icolor)
    
    # imcv = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    # hape = list(imcv.shape)
    # print(hape)
    
    if np.array_equal(pill(0,0), icolor):
        
        # print(icolor)
        # print(imcv[0,0])
        # print(1)
        print(imagePath[-8:],': correct')
        
        return 0
        
    elif np.array_equal(pill(0,-1), icolor):
        
        # print(icolor)
        # print(imcv[0,-1])
        print(imagePath, 2,'270')
        
        return 270
        
    elif np.array_equal(pill(-1,0), icolor):
            
        # print(icolor)
        # print(imcv[-1,0])
        print(imagePath, 3,'90')
        
        return 90
        
    elif np.array_equal(pill(-1,-1), icolor):
            
        # print(icolor)
        # print(imcv[-1,-1])
        print(imagePath, 4,'180')
        return 180
        
    else:
        
        
        print('?????????????')
        return 0

if __name__ == '__main__':
    dirr = 'images'
    lis = os.listdir(dirr)
    a = []
    for i in range(len(lis)-1, -1, -1):
        
        path = lis[i]
        imagePath = os.path.join(dirr, path)
        
        angle = check_rotate(imagePath)
        if angle != 0:
            a.append(path)
        
        
        '''
        if angle == 0:
            pass
        else:
            colorImage  = Image.open(imagePath)
            rotated = colorImage.rotate(angle, expand=True)
            rotated.save(imagePath) 
            
            
            angle = check_rotate(imagePath)'''
    

