import torch
from torch.utils.data import DataLoader
from dataloader.LanenetDataset import LaneDataset
from model.model import Lanenet
from utils.util import *
import argparse
import os
from tqdm import tqdm
from torch.autograd import Variable
from terminaltables import AsciiTable
from utils.transforms import *
from utils.data_augmentation import *



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./data/training_data/training",help=" Dataset_path")
    parser.add_argument("--save", required=False, default="./checkpoints", help="Directorty to save model checkpoint")
    parser.add_argument("--log", required=False, default="./log", help="Directory to save the log")
    parser.add_argument("--val", required=False, type=bool, default=True)
    parser.add_argument("--epoch", required=False, type=int, default=50, help="Training epoches")
    parser.add_argument("-b", "--batch_size", required=False, type=int, default=16, help="use validation")
    parser.add_argument("--lr", required=False, type=float, default=0.001, help="Learning rate")
    parser.add_argument("--pretrained", required=False, default="./save/ENet", help="pretrained model path")
    # parser.add_argument("--image", default="./output", help="output image folder")
    # parser.add_argument("--net", help="backbone network")
    parser.add_argument("--json", help="post processing json")

    return parser.parse_args()


def train(epoch):
    model.train()
    # progressbar = tqdm(enumerate(iter(train_loader)),leave = False,total = len(train_loader))
    for batch_idx,batch in enumerate(train_loader):
        optimizer.zero_grad()
        image_data = Variable(batch[0]).type(torch.FloatTensor).to(device)
        binary_label = Variable(batch[1]).type(torch.LongTensor).to(device)
        instance_label = Variable(batch[2]).type(torch.FloatTensor).to(device)

        net_output = model([image_data, binary_label, instance_label])
        ##if isinstance(model, torch.nn.DataParallel):
        batch_size = len(batch[0])
        # print(net_output['total_loss'].sum().item)
        total_loss = net_output['total_loss'].sum()
        instance_loss = net_output['instance_loss'].sum()
        binary_loss = net_output['instance_loss'].sum()
        iou = net_output['iou'].sum().item()/2  # the 2 is the gpu num
        prediction_num = net_output['prediction_num'].sum().item()/batch_size
        tp_num = net_output['TP_num'].sum().item()/batch_size
        total_loss.backward()
        optimizer.step()
        log_item = {'total_loss':total_loss.item(),'binary_loss':binary_loss.item(),'instance_loss':instance_loss.item(),'iou':iou,'prediction_num':prediction_num,'tp_num':tp_num}
        iter_idx = epoch *len(train_loader)+batch_idx
        logger.add_scalars('train',log_item,iter_idx)

        # print info to console
        if batch_idx % 2 == 0:
            table_data = [list(log_item.keys()), list(log_item.values())]
            table = AsciiTable(table_data).table
            print(f"Epoch:{epoch+1} | batch {batch_idx+1}/{len(train_loader)} \n"+table)



def val(epoch):
    print("val epoch:{}".format(epoch+1))
    model.eval()
    progressbar = tqdm(enumerate(iter(val_loader)), leave=False, total=len(val_loader))
    total_loss = 0
    binary_loss = 0
    instance_loss = 0
    iou = 0
    with torch.no_grad():
        for batch_idx, batch in progressbar:
            image_data = Variable(batch[0]).type(torch.FloatTensor).to(device)
            binary_label = Variable(batch[1]).type(torch.LongTensor).to(device)
            instance_label = Variable(batch[2]).type(torch.FloatTensor).to(device)

            net_output = model([image_data, binary_label, instance_label])
            # we have to print the loss
            batch_size = len(batch[0])
            total_loss += net_output['total_loss'].sum().item()
            binary_loss += net_output['binary_loss'].sum().item()
            instance_loss += net_output['instance_loss'].sum().item()
            iou += net_output['iou'].sum().item()
            # when batch_idx=0,select the whole batch to show the binary map
            if batch_idx == 0:
                # get the binary result and the instance result from gpu
                binary_pred = net_output['binary_pred'].detach().cpu()
                instance_result = net_output['instance_result'].detach().cpu()
                instance_result = minmax_scale(instance_result)
                size = list(binary_pred.size())
                size.insert(len(size), 3)
                # prepare the image
                binary = torch.zeros(size, dtype=torch.long)
                instance = instance_result[:, :3, :, :]
                binary[binary_pred == 1] = (255 * torch.ones(3, dtype=torch.long))
                binary = binary.permute(0, 3, 1, 2)
                logger.add_image("binary_map", binary, epoch)
                logger.add_image("instance_map", instance, epoch)
        total_loss = total_loss/len(val_loader)
        binary_loss = binary_loss/len(val_loader)
        instance_loss = instance_loss/len(val_loader)
        iou = iou/len(val_loader)
        print(f"total_loss:{total_loss} | binary_loss:{binary_loss} | instance_loss:{instance_loss} | iou:{iou}")
        logger.add_scalars("val",{'total_loss':total_loss,'binary_loss':binary_loss,'instance_loss':instance_loss,'iou':iou},step=epoch)



# step 1:get the param and handle
args = parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
logger = Logger(args.log)
if not os.path.isdir(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.log):
    os.makedirs(args.log)
train_dataset_file = os.path.join(args.dataset,"train.txt")
val_dataset_file = os.path.join(args.dataset,"val.txt")
# prepare the train and val dataset
# Imagenet mean, std
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
transform_train = Compose(Resize((512,256)), Rotation(2),
                          ToTensor(), Normalize(mean=mean, std=std))
train_dataset = LaneDataset(train_dataset_file,transform_train)
train_loader=DataLoader(train_dataset,batch_size = args.batch_size,shuffle = True)
if args.val:
    transform_val = Compose(Resize((512,256)), ToTensor(),
                            Normalize(mean=mean, std=std))
    val_dataset = LaneDataset(val_dataset_file,transform_val)
    val_loader = DataLoader(val_dataset,batch_size= args.batch_size,shuffle = True)

# step 3:load model and prepare the optimizer param
model = Lanenet()
model = torch.nn.DataParallel(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
start_epoch =0
# if pretrained,we have to load param
if args.pretrained:
    if 'enet' in args.pretrained.lower():
        load_Enet_pretrained(model, args.pretrained)
    else:
        print("please waiting,loading the pretrained parameters")
        start_epoch = load_model(args.pretrained, model, optimizer)
# train
print("All is prepared,be willing to train!")
for epoch in range(start_epoch,args.epoch):
    output = train(epoch)
    scheduler.step()
    if args.val:# and (epoch+1)%2==0:
        val_iou = val(epoch)
    if (epoch+1)%10==0:
        save_model(args.save,model,optimizer,epoch)
