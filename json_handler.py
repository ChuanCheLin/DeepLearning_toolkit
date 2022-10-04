#Author: Eric Lin
from itertools import count
import json
from json.encoder import INFINITY
from nis import cat
import os
import shutil
from typing import List
from tqdm import tqdm
from tabulate import tabulate
# name2id
# tea
# name2id = { 1:'brownblight', 2:'algal', 3: 'blister', 4: 'sunburn', 5: 'fungi_early', 6: 'roller',
#             7: 'moth', 8: 'tortrix', 9: 'flushworm', 10: 'caloptilia', 11: 'mosquito_early', 12: 'mosquito_late',
#             13: 'miner', 14: 'thrips', 15: 'tetrany', 16: 'formosa', 17: 'other', 18: 'nodicornis', 19: 'aleyrodidae', 20: 'termite', 21: 'inchmoth'}
# # cucumber
name2id = {1:'health', 2:'virus', 3:'anthracnose', 4:'downy', 5:'corynespora', 6:'powdery', 7:'malnutrition', 8:'leafminer'}

class json_handler():
    def __init__(self, jpg_data_root, coco_data_root, subset):
        self.full_jpg_dir = jpg_data_root #JPEGImages
        self.data_dir = coco_data_root + '/annotations/instances_' + subset + '.json' #json file
        self.txt_dir = coco_data_root + '/' + subset + '.txt' #txt file
        self.new_jpg_dir = coco_data_root + '/' + subset #train val trainval......
        self.coco_data_root = coco_data_root
        self.subset = subset

    def check_info(self):
        
        #open file
        j = open(self.data_dir)
        
        # load info in json
        all_info = json.load(j)
        # print((all_info).keys()) # dict_keys(['images', 'type', 'annotations', 'categories'])
        images = all_info['images']
        annotations = all_info['annotations']
        categories = all_info['categories']
        
        categories_count = []
        categorie_names = []
        for cat in categories:
            categories_count.append(0)
            categorie_names.append(cat['name'])
        
        for anno in annotations:
            categories_count[anno['category_id']-1] += 1

        all_data = []
        for i in range(len(categorie_names)):
            all_data.append([categorie_names[i], categories_count[i]])
        
        headers = ['category', '#instances']

        print(tabulate(all_data, headers, tablefmt="grid"))

    #  get jpg list from arbitrary json file & save as a txt file
    def write_jpg_txt(self):
        
        #open file
        j = open(self.data_dir)
        
        # load info in json
        all_info = json.load(j)
        images = all_info['images']

        # write txt file
        file = open(self.txt_dir, 'w')       

        # get value
        for i in range(len(images)):
            file.write(images[i].get('file_name'))
            file.write('\n')

        file.close()
    
    # copy jpg file into corresponding coco dataset dir
    def get_jpg_from_txt(self, single_class = False):

        if single_class:
            if(os.path.isdir(self.single_class_jpg_dir)==False):
                os.mkdir(self.single_class_jpg_dir)
            f = open(self.single_class_txt_dir, 'r')
            for jpg in tqdm(f.readlines()):
                jpg = jpg.rstrip('\n')
                ori_path = self.full_jpg_dir + jpg
                new_path = self.single_class_jpg_dir +'/' + jpg
                shutil.copyfile(ori_path, new_path)

            print('jpg files saved at ' + str(self.single_class_jpg_dir))
        else:
            if(os.path.isdir(self.new_jpg_dir)==False):
                os.mkdir(self.new_jpg_dir)

            f = open(self.txt_dir, 'r')
            for jpg in tqdm(f.readlines()):
                jpg = jpg.rstrip('\n')
                ori_path = self.full_jpg_dir + jpg
                new_path = self.new_jpg_dir +'/' + jpg
                shutil.copyfile(ori_path, new_path)

    #  get jpg list from a specific class
    def write_single_class_txt(self, target_class_id = int):
        
        #open file
        j = open(self.data_dir)
        
        # load info in json
        all_info = json.load(j)
        images = all_info['images']
        annotations = all_info['annotations']
        image_list = []

        # get img list that contains specific class from annotations
        for i in range(len(annotations)):
            id = annotations[i].get('category_id')
            image_id = annotations[i].get('image_id')
            if(id == target_class_id and image_id not in image_list):  
                image_list.append(image_id)

        # write txt file
        dir = self.coco_data_root + '/' + self.subset + '_' + name2id.get(target_class_id) + '.txt'
        self.single_class_jpg_dir = self.coco_data_root + '/' + self.subset + '_' + name2id.get(target_class_id)
        self.single_class_txt_dir = dir
        print('txt file saved at ' + str(dir))
        file = open(dir, 'w') 

        for i in range(len(images)):
            if(images[i].get('id') in image_list):
                file.write(images[i].get('file_name'))
                file.write('\n')

    def create_dataset_with_selected_classes(self, selected = list, output_jsonpath = str, type = str, thresh = int):
        
        #open file
        j = open(self.data_dir)
        
        # load info in json
        all_info = json.load(j)
        # print((all_info).keys()) # dict_keys(['images', 'type', 'annotations', 'categories'])

        output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
        }

        images = all_info['images']
        annotations = all_info['annotations']
        categories = all_info['categories']

        selected_image_set = set()
        
        if type == 'few':
            count = []
            for ctg in categories:
                count.append(0)
        else:
            thresh = INFINITY
        
        for anno in annotations:
            if anno['category_id'] in selected and count[anno['category_id']-1] < thresh:
                if anno['image_id'] not in selected_image_set:
                    selected_image_set.add(anno['image_id'])
                    count[anno['category_id']-1] += 1
        print(count)

        for anno in annotations:
            if anno['image_id'] in selected_image_set:
                output_json_dict['annotations'].append(anno)

        for img in images:
            if img['id'] in selected_image_set:
                output_json_dict['images'].append(img)

        for ctg in categories:
            # if ctg['id'] in selected:
            output_json_dict['categories'].append(ctg)

        selected_ctg_name = []
        for key in selected:
            selected_ctg_name.append(name2id.get(key))


        print(f'selected categories: {selected_ctg_name}')
        print(f'selected images: {len(selected_image_set)}')
        print(f'selected annotations: {len(output_json_dict["annotations"])}')

        with open(output_jsonpath, 'w') as f:
            output_json = json.dumps(output_json_dict)
            f.write(output_json)


        

  
if __name__ == "__main__":  
    tea = json_handler(
        jpg_data_root= "/home/eric/mmdetection/data/VOCdevkit/datasets/VOC2007/JPEGImages/",
        coco_data_root = "/home/eric/mmdetection/data/VOCdevkit/datasets/set1/split4/few60", subset = 'test')
    
    cucumber = json_handler(
        jpg_data_root= "/home/eric/mmdetection/data/VOCdevkit/datasets/cucumber/trainval/",
        coco_data_root = "/home/eric/mmdetection/data/VOCdevkit/datasets/cucumber/", subset = 'trainval')
    
    ###
    set1_base = [1, 2, 3, 4, 6, 7]
    set1_few = [5, 8]
    ###

    # cucumber.check_info()
    cucumber.create_dataset_with_selected_classes(selected=set1_few, 
    output_jsonpath ="/home/eric/mmdetection/data/VOCdevkit/datasets/cucumber/annotations/split1_few.json",
    type = 'few', thresh = 60)

    # tea.check_info()

    # tea.write_jpg_txt()
    # tea.get_jpg_from_txt()

    # tea.write_single_class_txt(19)
    # tea.get_jpg_from_txt(single_class=True)


