import torchvision.transforms as trans
from utils.transforms import *
from utils.data_augmentation import *
import cv2
import os.path as ops
import time
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
# giu_ids = [0,1]
# print('cuda:{}'.format(giu_ids[0]))
# device = torch.device('cuda:{}'.format(giu_ids[0]) if torch.cuda.is_available() else 'cpu')
from utils.transforms import *
from model.model import *
from utils.util import *
import torch.nn.functional as F
from utils.postprocess import LaneNetPostProcessor
postprocess = LaneNetPostProcessor()
import matplotlib.pyplot as plt
def test_lanenet(image_path,weights_path):
    '''

    :param image_path:
    :param weights_path:
    :return:
    '''

    assert ops.exists(image_path), '{:s} not exist'.format(image_path)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Lanenet().to(device)
    load_model(weights_path, model)
    print('Start reading image and preprocessing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = (torch.from_numpy(image) / 255.)
    print(image.size())
    # image = F.normalize()
    image = (image-mean) / std
    image = image.permute(2,0,1).unsqueeze(0)
    print('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))
    t_val_start=time.time()
    image_data = image.type(torch.FloatTensor).to(device)
    net_output = model([image_data])
    binary_pred = net_output['binary_pred'].detach().cpu()[0].numpy()
    instance_result = net_output['instance_result'].detach().cpu()[0].permute(1,2,0).numpy()
    # instance_result = minmax_scale(instance_result).squeeze(0).permute(1,2,0).numpy().astype(np.uint8)

    ret = postprocess.postprocess(binary_pred,instance_result,source_image=image_vis,plot=True)
    print(ret.keys())
    print(ret['pred_points'])
    mask_image = ret['mask_image']
    handled_image = ret['source_image']
    # instance_view = minmax_scale(instance_result).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)[0]
    print(ret["fit_params"])
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(mask_image[:, :, (2, 1, 0)])
    plt.title('mask_image')
    plt.subplot(1,3,2)
    plt.imshow(image_vis[:, :, (2, 1, 0)])
    plt.title('src_image')
    # plt.subplot(2,2,3)
    # instance_result = instance_result.astype(np.uint8)
    # plt.imshow(instance_result[:,:,(2,1,0)])
    # plt.title('instance_image')
    plt.subplot(1,3,3)
    plt.imshow(binary_pred* 255, cmap='gray')
    plt.title('binary_image')
    plt.show()
    plt.pause(0)

    # print(instance_result.shape)
    # binary_view = np.zeros((256,512,3),dtype = np.uint8)
    # binary_view[binary_pred == 1] =[255,255,255]
    # cv2.imshow("binary", binary_view)
    # # cv2.imshow("instance",instance_result[:,:,:3])
    # cv2.waitKey()

    # # prepare the image
    # binary = torch.zeros(size, dtype=torch.long)
    # instance = instance_result[:, :3, :, :]
    # print(size)
    # size.insert(len(size), 3)
    # # prepare the image
    # binary = torch.zeros(size, dtype=torch.long)
    # instance = instance_result[:, :3, :, :]
    # binary[binary_pred == 1] = (255 * torch.ones(3, dtype=torch.long))
    # binary = binary.permute(0, 3, 1, 2)

if __name__ == '__main__':
    test_lanenet("G:/learningData/tusimple/clips/0530/1492637052323829842_0/20.jpg","checkpoints/lanenet_epoch_19.pth")
# import torch
# from tensorboardX import SummaryWriter
# import numpy as np
# import cv2
# from utils.util import *
# save_name = "./log/save"
# logger = Logger("./log","save")
# tensor = np.ones((720,1280,3),dtype=np.uint8)*255
# tensor2 = cv2.imread("./data/training_data_example/gt_image_binary/0000.png")
# tensor2[:,:,:]=[255,255,255]
# print(tensor==tensor2)
# print(tensor.shape)
# print((tensor[:,:,0]==255).sum())

# print(tensor[400:420,500:600,:])
# # tensor[:,:,0] = 255
# # tensor[1,:,:] = 255
# cv2.imshow("nnn",tensor)
# cv2.waitKey()
# import torch
# from torchvision.utils import make_grid
# binary_pred = torch.LongTensor(8,256,512).random_() % 2
# print(binary_pred)
# size = list(binary_pred.size())
# size.insert(len(size), 3)
# # prepare the image
# binary = torch.ones(size, dtype=torch.long) * 255
# # binary_pred[:,:,:,:] = (255 * torch.ones(3, dtype=torch.long))
# binary = binary.permute(0, 3, 1, 2)
# print(binary)
# logger.add_image("test",binary,1)