import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
from torchvision.utils import make_grid
def save_model(save_path,model,optimizer,epoch):
    assert os.path.exists(save_path)
    save_name = os.path.join(save_path,f"lanenet_epoch_{epoch}.pth")
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimzer_state_dict':optimizer.state_dict()

    },save_name)
    print(f"epoch{epoch+1}:the model is saved")
def load_model(path,model,optimizer):
    assert os.path.exists(path)
    save_dict = torch.load(path)
    epoch = save_dict['epoch'+1]
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(save_dict['model_state_dict'])
    else:
        model.load_state_dict(save_dict['model_state_dict'])
    optimizer.load_sate_dict(save_dict['optimzer_state_dict'])
    return epoch

class Logger(object):

    def __init__(self,path,logname =None):
        if not os.path.exists(path):
            os.makedirs(path)
        if logname:
            save_name = os.path.join(path,logname)
        else:
            timestamp = datetime.fromtimestamp(time.time()).strftime('%m%d-%H:%M')
            save_name = os.path.join(path,timestamp)
        self.writer = SummaryWriter(save_name)

    def add_scalars(self,flag,tag_values_pairs,step):

        for tag,value in tag_values_pairs.items():
            self.writer.add_scalar(flag+'/'+tag,value,step)

    def add_image(self,tag,tensor,step):
        make_grid(tensor)
        self.add_image(tag,tensor,step,dataformats="HWC")
