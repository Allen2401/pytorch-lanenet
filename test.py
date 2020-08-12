import torchvision.transforms as trans
from utils.transforms import *
from utils.data_augmentation import *
import cv2
from model.model import *
import torch
from  copy import deepcopy
from dataloader.LanenetDataset import LaneDataset
from torch.utils.data import DataLoader
from utils.util import Logger
from torchvision.utils import make_grid
# img = cv2.imread("data/training_data_example/image/0000.png",cv2.IMREAD_COLOR)
# mean=(0.485, 0.456, 0.406)
# transform_val = Compose(Resize((512,256)), Rotation(2),ToTensor(), Normalize(mean=mean, std=std))
# # Normalize(mean=mean, std=std)
# val_dataset = LaneDataset("data/training_data_example/val.txt",transform_val)
# img = val_dataset[0]['img'].numpy().transpose(1,2,0)
# cv2.imshow('image',img)
# a = cv2.imwrite('data/training_data_example/image/0000xx.png',img)
# cv2.waitKey()
# giu_ids = [0,1]
# print('cuda:{}'.format(giu_ids[0]))
# device = torch.device('cuda:{}'.format(giu_ids[0]) if torch.cuda.is_available() else 'cpu')
# logger = Logger("./log",'test')
# logger.add_scalars('train',{'zh':1,'zhj':2},1)
# train_dataset = LaneDataset('data/training_data_example/train.txt')
# train_loader = DataLoader(train_dataset,batch_size=4)
# for batch_idx, batch in enumerate(train_loader):
#     ##if isinstance(model, torch.nn.DataParallel):
#     print(len(batch))
#     print(len(batch[0]))
# #     # batch_size = len(batch[0])
# import numpy as np
# binary_pred = torch.rand(8,2,256,256)
# binary_pred = torch.argmax(binary_pred,dim=1)
# print(binary_pred.size())
# print(binary_pred)
# print(binary_pred.shape)
# size = list(binary_pred.shape)
# # n*h*w*3
# size.insert(len(size),3)
# binary = np.zeros(size)
# print(binary.shape)
# # binary_pred is n*h*w
# binary[binary_pred==1]=[255,255,225]
# display_imgs = []
# cv2.imshow("name",binary[0])
# cv2.waitKey()
# for i in range(8):
#     print(binary[i].shape)
# #     display_imgs.append(cv2.cvtColor(binary[i],cv2.COLOR_BGR2RGB))
# tensor = torch.randint(0,256,(16,3,512,256))
# make_grid(tensor)
model = Lanenet()
a = 0
# encoder 一共有329层
enet_state_dict = torch.load("./save/ENet")['state_dict']
for index,name in enumerate(model.state_dict()):
  print(name)
  if 'encoder' in name and index <329:
      enetname = name.replace('encoder.','')
      model.state_dict()[name].copy_(enet_state_dict[enetname])
  elif 'decoder' in name:
      if 'transposed' in name:
          continue
      head = ''
      if 'binary' in name:
          head ='decoder.binary_'
      else:
          head = 'decoder.instance_'
      enetname = name.replace(head,'')
      model.state_dict()[name].copy_(enet_state_dict[enetname])
  else:
      print("there must have something wrong!")
      break
  # print(a)
  # print(model.state_dict()[name])
# a = torch.load("./save/ENet")
# print(list(a['state_dict'].keys()))
