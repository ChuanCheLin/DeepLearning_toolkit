#encoding:utf-8
#
#created by xiongzihua
#

import sys
import os
import matplotlib.pyplot as plt 
import cv2
import numpy as np



Color = [   [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128]]
        
def parse_det(detfile):
    '''
    Parameters
    ----------
    detfile : txt file
        YOLO detection result

    Returns
    -------
    result : list
        format: [(x1,y1),(x2,y2),cls,prob] for each detect
    '''
    result = []
    with open(detfile, 'r') as f:
        for line in f:
            token = line.strip().split()
            if len(token) != 10:
                continue
            x1 = int(float(token[0]))
            y1 = int(float(token[1]))
            x2 = int(float(token[4]))
            y2 = int(float(token[5]))
            cls = token[8]
            prob = float(token[9])
            result.append([(x1,y1),(x2,y2),cls,prob])
    return result 



def gt_visualize(item, classes):
    '''
    item: (img, bbox, label(number), score)
    img
    '''
    scr = False
    image = np.array(item[0])
    image = (image[0]+1)/ 2 * 255
    image = np.swapaxes(image,0,2)
    image = image.astype(np.uint8)
    image = image.copy()
#    print(image[0])
    size = image.shape
#    print(type(image), size)
    bbox = np.array(item[1])
    label = np.array(item[2])
    if len(item) == 4:
        score = np.array(item[3])
        scr = True
    count = label.shape[1]
#    print(count)
    for i in range(count):
#        print(label,type(label),label[0,i],i)
        class_name = classes[label[0,i]]
#        print(class_name)
#        print(bbox,bbox.shape)
        xywh = bbox[0,i,:]        
        xywh = xywh.flatten()
#        print(xywh)
        w = int((xywh[3])*size[0])
        h = int((xywh[2])*size[1])
        x = int((xywh[1])*size[0] - w/2 )
        y = int((xywh[0])*size[1] - h/2 )
#        print(x, y, w,h)
        color = Color[int(label[0,i])]
#        print(x,y,x+w,y+h,color)
        cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
        text_size, baseline = cv2.getTextSize(class_name,
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (x, y - text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline),
                      (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
#        cv2.imwrite('test2.jpg',image)
        cv2.putText(image, class_name, (p1[0], p1[1] + baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
        cv2.imwrite('test3.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    return image

def pred_visualize(img, pred, classes):
    '''
    item: (img, bbox, label(number), score)
    img
    '''
    image = np.array(img)
    image = np.swapaxes(image,0,2)
    image = image.astype(np.uint8)
    image = image.copy()
#    print(image[0])
    size = image.shape
#    print(type(image), size)
    count = label.shape[1]
#    print(count)
    for i in range(count):
#        print(label,type(label),label[0,i],i)
        class_name = classes[label[0,i]]
#        print(class_name)
#        print(bbox,bbox.shape)
        xywh = bbox[0,i,:]        
        xywh = xywh.flatten()
#        print(xywh)
        w = int((xywh[3])*size[0])
        h = int((xywh[2])*size[1])
        x = int((xywh[1])*size[0] - w/2 )
        y = int((xywh[0])*size[1] - h/2 )
#        print(x, y, w,h)
        color = Color[int(label[0,i])]
#        print(x,y,x+w,y+h,color)
        cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
        text_size, baseline = cv2.getTextSize(class_name,
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (x, y - text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline),
                      (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
#        cv2.imwrite('test2.jpg',image)
        cv2.putText(image, class_name, (p1[0], p1[1] + baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
        cv2.imwrite('test3.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    return image

if __name__ == '__main__':
    imgfile = sys.argv[1]
    detfile = sys.argv[2]

    image = cv2.imread(imgfile)
    result = parse_det(detfile)
    for left_up,right_bottom,class_name,prob in result:
        color = Color[DOTA_CLASSES.index(class_name)]
        cv2.rectangle(image,left_up,right_bottom,color,2)
        label = class_name+str(round(prob,2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), 
                      (p1[0] + text_size[0], p1[1] + text_size[1]),
                      color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imwrite('result.jpg',image)



