import torchvision.transforms as trans
from utils.transforms import *
from utils.data_augmentation import *
import cv2
import torch
from dataloader.LanenetDataset import LaneDataset
# img = cv2.imread("data/training_data_example/image/0000.png",cv2.IMREAD_COLOR)
# mean=(0.485, 0.456, 0.406)
# std=(0.229, 0.224, 0.225)
# transform_val = Compose(Resize((512,256)), Rotation(2),ToTensor(), Normalize(mean=mean, std=std))
# # Normalize(mean=mean, std=std)
# val_dataset = LaneDataset("data/training_data_example/val.txt",transform_val)
# img = val_dataset[0]['img'].numpy().transpose(1,2,0)
# cv2.imshow('image',img)
# a = cv2.imwrite('data/training_data_example/image/0000xx.png',img)
# cv2.waitKey()
giu_ids = [0,1]
print('cuda:{}'.format(giu_ids[0]))
device = torch.device('cuda:{}'.format(giu_ids[0]) if torch.cuda.is_available() else 'cpu')