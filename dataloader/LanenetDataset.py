from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms
import os
import torch
import cv2
import numpy as np

class LaneDataset(Dataset):

    def __init__(self,dataset,transform = None):
        self._gt_image_list = []
        self._gt_label_binary_list = []
        self._gt_label_instance_list = []
        self.transform = transform

        with open(dataset,"r")as file:
            for line in file:
                line = line.strip(" \n").split(" ")
                self._gt_image_list.append(line[0])
                self._gt_label_binary_list.append(line[1])
                self._gt_label_instance_list.append(line[2])
            assert len(self._gt_image_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

    def __len__(self):
        return len(self._gt_image_list)

    def __getitem__(self, item):
        img = cv2.imread(self._gt_image_list[item],cv2.IMREAD_COLOR)
        label_binary_img = cv2.imread(self._gt_label_binary_list[item],cv2.IMREAD_UNCHANGED)
        label_instance_img = cv2.imread(self._gt_label_instance_list[item],cv2.IMREAD_UNCHANGED)
        label_binary_img[np.where(label_binary_img == 255)] = 1
        sample = {'img':img,'binary_label':label_binary_img,'instance_label':label_instance_img}
        if self.transform is not None:
            sample = self.transform(sample)
        # trans = transforms.Compose(transforms.Resize((512,256)))
        # label_binary_img = trans(label_binary_img)
        # label_instance_img = trans(label_instance_img)
        # img = img.reshape(img.shape[2],img.shape[0],img.shape[1])
        # label_binary_img[np.where(label_binary_img==255)]=1
        # print(sample['instance_label'].type())
        return sample['img'],sample['binary_label'],sample['instance_label']