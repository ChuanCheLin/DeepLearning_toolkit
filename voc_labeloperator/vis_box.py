# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:35:33 2020

@author: Chun
"""


import sys
import os
import matplotlib.pyplot as plt 
import mmcv
import csv
import cv2
import numpy as np
from pascal_voc_io import PascalVocReader as PR
from mmcv.image import imread, imwrite

classes = ['','brownblight', 'blister', 'algal',  'fungi_early',
                 'miner',   'thrips',
                 'mosquito_early', 'mosquito_late',
                 'moth', 'tortrix',   'flushworm',
                 'roller',   'other']

classesnew =['mos_e','mos_l','brown', 'fungi','blister','algal','miner','thrips','ori_w','tor_w','tor_r', 'flush','']

table = {
        'mosquito_early': 'mos_e',
        'mosquito_late':'mos_l',
        'brownblight': 'brown',
        'fungi_early': 'fungi',
        'blister': 'blister',
        'algal': 'algal',
        'miner': 'miner',
        'thrips':'thrips',
        'roller': 'ori_w',
        'moth': 'tor_w',
        'tortrix': 'tor_r',
        'flushworm': 'flush'
    }
colortable = {
'mos_e': (48, 27, 155), 
'mos_l': (0, 102, 204), 
'brown': (0, 0, 128), 
'fungi': (255, 102, 102), 
'blister': (0, 128, 0), 
'algal': (0, 128, 128), 
'miner': (128, 0, 0), 
'thrips': (128, 0, 128), 
'ori_w': (128, 128, 0), 
'tor_w': (128, 0, 64), 
'tor_r': (128, 128, 64), 
'flush': (128, 0, 192), 
'other': (0, 0, 0)
}

def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)

def parse_xml(file):
    '''
    Parameters
    ----------
    detfile : xml file
        VOC gt.xml

    Returns
    -------
    result : list
        format: [ (x1,y1),(x2,y2),cls,prob ] for each detect (yolo)
        format: [ (x1,y1),(x2,y2),cls ] for each detect (voc)
    '''
    pr = PR(file)
    shapes = pr.getShapes()
    result = []
    for shape in shapes:
        result.append([shape[1][0], shape[1][2], shape[0]])
    return result


def vis_gtbox(img_path,
              gt_boxes,
              colors,
              csvw=None,
              width=None,
              class_names=None,
              thickness=1,
              font_scale=0.5,
              show=True,
              win_name='',
              out_file=None,
              flip=False):
    '''
    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    bboxes : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.
    class_names : TYPE, optional
        DESCRIPTION. The default is class_names.
    show : TYPE, optional
        DESCRIPTION. The default is show.
    wait_time : TYPE, optional
        DESCRIPTION. The default is wait_time.
    out_file : TYPE, optional
        DESCRIPTION. The default is out_file.

    Returns
    -------
    None.

    '''
    
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ori_size = img.shape
    
    
    if width == None:
        ratio = 1.0
    else:
        ratio = width/ori_size[0]
        img = cv2.resize(img, (int(ori_size[1]*ratio),int(ori_size[0]*ratio)))
    
    
    ABC = 'ABCDEFGHIJKLMNOP'
    i = 0
    '''
    d = {'mosquito_early' 155,27,48 orange
        'mosquito_late' 204,102,0 suguralmond
        'brownblight' 128,0,0 aa
        'fungi_early' 102,102,255 purpleblue
        'blister' 0,128,0 aa
        'algal' 128,128,0 aa
        'miner' 0,0,128 aa
        'thrips' 128,0,128 aa
        'roller' 0,128,128 aa
        'moth' 64,0,128 aa
        'tortrix' 64,128,128 aa
        'flushworm' 192,0,128 aa
        'other' 0,0,0 vvv
         }
    '''
    csvw = []
    dpi = 200
    fig = plt.figure(frameon=False)
    fs = int(img.shape[0]/100)
    lw = int(img.shape[0]/300)
    box_alpha = 0.7
    fig.set_size_inches(img.shape[1] / dpi, img.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(img)
    
    for gt_box in gt_boxes:
        
        label = gt_box[2]
        
        if label == '':
            continue
        
        left_top = np.array(gt_box[0])
        right_bottom = np.array(gt_box[1])
        
        left_top = left_top*ratio
        right_bottom = right_bottom*ratio
        
        left_top = left_top.astype(np.int32)
        right_bottom = right_bottom.astype(np.int32)
        
        #color = colors[classesnew.index(label)]
        color = color_val_matplotlib(colortable.get(label))
        
        box_id = label
        # print(color.dtype)
        
        
        ax.add_patch(
            plt.Rectangle((gt_box[0]),
                          right_bottom[0] - left_top[0],
                          right_bottom[1] - left_top[1],
                          fill=False, edgecolor=tuple(color),
                          linewidth=lw, alpha=box_alpha))
                          
        ax.text(
                left_top[0], left_top[1] - 2,
                label,
                fontsize=fs,
                family='serif',
                bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            }, 
                color='white')
        '''
        
        cv2.rectangle(img, (left_top[0], left_top[1]),
                      (right_bottom[0], right_bottom[1]), color, 8)
        text_size, baseline = cv2.getTextSize(box_id,
                                              cv2.FONT_HERSHEY_SIMPLEX, t_size, int(t_size))
        p1 = (left_top[0], left_top[1] + text_size[1])
        cv2.rectangle(img, tuple(left_top), (p1[0] + text_size[0], p1[1]+15 ), color, -1)
        cv2.putText(img, box_id, (p1[0], p1[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, t_size, (255,255,255), int(t_size)*2, 8)
        
        csvw.append([img_path[-8:], box_id, label,
                     left_top[0], left_top[1], right_bottom[0], right_bottom[1]])
            '''
        i += 1
    
    # if flip:
    #     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if out_file is not None:
        fig.savefig(out_file, dpi=dpi)
        plt.close('all')
        # cv2.imwrite(out_file, img)    
        print('done   '+ str(out_file))
        
    with open('table.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csvw)
        
    return 
    
    
    
def vis_predbox(img,
                result,
                class_names,
                score_thr=0.3,
                wait_time=0,
                show=True,
                out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    
    bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    
    # draw bounding boxes
    draw_det_bboxes_A(
        img,
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=show,
        wait_time=wait_time,
        out_file=out_file)
    if not (show or out_file):
        return img

def draw_det_bboxes_A(img_name,
                        bboxes,
                        labels,
                        Pred,
                        width=800,
                        class_names=None,
                        score_thr=0,
                        out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    
    img = imread(img_name,cv2.IMREAD_UNCHANGED)
    img = img.copy()
    
    ori_size = img.shape
    ratio = width/ori_size[0]
    img = cv2.resize(img, (int(ori_size[1]*ratio),int(ori_size[0]*ratio)))
    
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        scores = scores[inds]

    ABC = 'ABCDEFGHIJKLMNOP'
    i = 0
    
    for bbox, label in zip(bboxes, labels):
        
        class_name = class_names[label]
        color = colors[label]
        box_id = ABC[i]
        
        bbox = bbox*ratio
        bbox_int = bbox.astype(np.int32)
        
        write_det(Pred,
                  box_id,
                  pred_cls,
                  score,
                  bbox_int)
        
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        
        cv2.rectangle(img, (left_top[0], left_top[1]),
                      (right_bottom[0], right_bottom[1]), color, 8)
        text_size, baseline = cv2.getTextSize(box_id,
                                              cv2.FONT_HERSHEY_SIMPLEX, t_size, 5)
        p1 = (left_top[0], left_top[1] + text_size[1])
        cv2.rectangle(img, tuple(left_top), (p1[0] + text_size[0], p1[1]+15 ), color, -1)
        cv2.putText(img, box_id, (p1[0], p1[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, t_size, (255,255,255), 5, 8)
        
        i += 1
        
        
    print('done   '+ str(out_file))
    
    if out_file is not None:
        imwrite(img, out_file)




def read_label_color(file, BGR=True):
    '''
        Parameters
        ----------
        file : txt file
            format: brownblight 255,102,0 orange
    
        Returns
        -------
        result : dict
            format: [ 'brownblight' : [255, 102, 0]]
    '''
    color_dict = {}
    
    with open(file, 'r') as f:
        lines = f.readlines()
    
    for l in lines:
        strs = l.split(' ')
        label = strs[0]
        color = list(map(int, strs[1].split(',')))
        if BGR:
            color[0], color[2] = color[2], color[0]
        color_dict[label] = color
        
    return color_dict
    
    
def make_csv(csvw):
    
    with open('table.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csvw)


if __name__ == '__main__':
    color_list = [[0.5, 0, 0],
                [0, 0.5, 0],
                [0.5, 0.5, 0],
                [0, 0, 0.5],
                [0.5, 0, 0.5],
                [0, 0.5, 0.58],
                [0.5, 0.5, 0.5],
                [0.25, 0, 0],
                [0.75, 0, 0],
                [0.25, 0.5, 0],
                [0.75, 0.5, 0],
                [0.25, 0, 0.5],
                [0.75, 0, 0.5],
                [0.25, 0.5, 0.5],
                [0.75, 0.5, 0.5],
                [0, 0.25, 0],
                [0.5, 0.25, 0],
                [0, 0.75, 0],
                [0.5, 0.75, 0],
                [0, 0.25, 0.5]]
    img_dir = "/home/eric/mmdetection/data/VOCdevkit/VOC2007/JPEGImages/"
    # img_list = os.listdir(img_dir)
    im_list = []
    testfile = "/home/eric/mmdetection/data/VOCdevkit/VOC2007/ImageSets/Main/draw_gt.txt"
    f1 = open(testfile, 'r')
    testlis = f1.read().splitlines()
    f1.close()
    for tf in testlis:
        im_list.append( tf+'.jpg' )

    
    colorfile = 'color.txt'
    colors = read_label_color(colorfile)
    #print(colors)
    ext_lis = os.listdir('tempan')
    
    for img in im_list:
        if img in ext_lis:
            continue
        img_path = os.path.join(img_dir, img) 
        
        _xml = '/home/eric/mmdetection/data/VOCdevkit/VOC2007/Annotations/' + img[:4] + '.xml'
        re = parse_xml(_xml)
        
        
        for i in range(len(re)):
            if re[i][2] == 'stick' or re[i][2] == 'other' or re[i][2] == 'dew':
                re[i][2] = ''
            re[i][2] = table.get(re[i][2])

            
        vis_gtbox(img_path,
                  re,
                  color_list,
                  csvw=None,
                  thickness=1,
                  font_scale=0.5,
                  show=True,
                  win_name='',
                  out_file='tempan/' + img)

