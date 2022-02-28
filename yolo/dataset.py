import torch
import os
import pandas as pd
import cv2

from utils.box_processes import BoxProcesses


# TODO: Move bbox -> cellbox function to preprocesses.py

'''
Class for pytorch data loader
# Dataset converts absolute bbox into cellbox bbox when loading. 
'''
class VOCDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file, header='infer') # img, label dataframe
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform # callback function
        self.box_processes = BoxProcesses(S, B, C)


    def __len__(self):
        '''
        Get number of samples
        '''
        return len(self.annotations)  


    def __getitem__(self, index):
        '''
        Get one sample
        return: 7x7x25 shape label.
        label format is class, x, y, w, h within 0~1 value
        (different from raw PascalVOC format)
        '''
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        # read boxes information from file
        with open(label_path) as f:
            for i, label in enumerate(f.readlines()):
                label = label.replace("\n", "").split()
                class_label = int(label[0])
                x = float(label[1])
                y = float(label[2])
                width = float(label[3])
                height = float(label[4])

                boxes.append([class_label, x, y, width, height])
    
        # read image and initialize tensor
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # if preprocessing is defined
        if self.transform:
            image, boxes = self.transform(image, boxes)

        # convert absolute box to cell boxes
        boxes = torch.tensor(boxes)
        label_matrix = self.box_processes.boxes_absolute_to_cell(boxes)

        return image, label_matrix # x, y