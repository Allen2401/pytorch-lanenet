import os
import torch
from tensorboardX import SummaryWriter
from datetime import datetime
import time
from torchvision.utils import make_grid
import numpy as np
def save_model(save_path,model,optimizer,epoch):
    assert os.path.exists(save_path)
    save_name = os.path.join(save_path,f"lanenet_epoch_{epoch}.pth")
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimzer_state_dict':optimizer.state_dict() if optimizer is not None else None
    },save_name)
    print(f"epoch{epoch+1}:the model is saved")
def load_model(path,model,optimizer=None):
    assert os.path.exists(path)
    save_dict = torch.load(path)
    epoch = save_dict['epoch']+1
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(save_dict['model_state_dict'])
    else:
        model.load_state_dict(save_dict['model_state_dict'])
    # if save_dict['optimzer_state_dict'] is not None:
    #     optimizer.load_sate_dict(save_dict['optimzer_state_dict'])
    return epoch

def load_Enet_pretrained(model,filepath):
    enet_state_dict = torch.load(filepath)['state_dict']
    for index, name in enumerate(model.state_dict()):
        print(name)
        if 'encoder' in name and index < 329:
            if isinstance(model,torch.nn.DataParallel):
                enetname = name.replace('module.encoder.', '')
            else:
                enetname = name.replace('encoder.', '')
            model.state_dict()[name].copy_(enet_state_dict[enetname])
        elif 'decoder' in name:
            if 'transposed' in name:
                continue
            head = ''
            if 'binary' in name:
                head = 'decoder.binary_'
            else:
                head = 'decoder.instance_'
            if isinstance(model, torch.nn.DataParallel):
                head = 'module.' + head
            enetname = name.replace(head, '')
            model.state_dict()[name].copy_(enet_state_dict[enetname])
        else:
            print("there must have something wrong!")
            break
    # we best to save the param
    save_model('./save', model, optimizer=None,epoch=-1)


def minmax_scale(input_arr):
    # the function is for nomalization ,and the funciton also changes the dim
    b,c,h,w= input_arr.size()
    input_arr = input_arr.reshape(b,c,-1)
    max_val = torch.max(input_arr,2,keepdim=True)[0]
    min_val = torch.min(input_arr,2,keepdim=True)[0]
    output_arr = (input_arr - min_val)*255.0 / (max_val - min_val)
    output_arr = output_arr.reshape(b,c,h,w)
    return output_arr



class Logger(object):

    def __init__(self,path,logname =None):
        if not os.path.exists(path):
            os.makedirs(path)
        if logname:
            save_name = os.path.join(path,logname)
        else:
            timestamp = datetime.fromtimestamp(time.time()).strftime('%m%d-%H-%M')
            save_name = os.path.join(path,timestamp)
        self.writer = SummaryWriter(save_name)

    def add_scalars(self,flag,tag_values_pairs,step):

        for tag,value in tag_values_pairs.items():
            self.writer.add_scalar(flag+'/'+tag,value,step)

    def add_image(self,tag,tensor,step):
        tensor = make_grid(tensor,4)
        to_num = tensor.numpy().astype(np.uint8)
        self.writer.add_image(tag,to_num,step,dataformats='CHW')

def polyLR(optimizer,epoch,lr,power):
    lr = lr * pow((1-1.0*epoch/100),power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr