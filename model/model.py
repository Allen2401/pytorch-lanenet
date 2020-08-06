from model.encoder import *
from model.decoder import *
import torch
import torch.nn as nn
from model.loss import *

class Lanenet(nn.Module):

    def __init__(self,instance_depth=5):

        super(Lanenet,self).__init__()
        self.encoder = EnetEncoder()
        self.decoder = EnetDecoder(instance_depth)

    def forward(self,input):
        forward_input = input[0]
        binary_label = input[1]
        instance_label = input[2]
        batch_size = forward_input.size(0)
        encoder_result,indices = self.encoder(forward_input)
        binary_result,instance_result = self.decoder(encoder_result,indices)
        # after get the result ,we have to compute the loss
        binary_loss = weighted_cross_entropy_loss(binary_result, binary_label)
        # the instance branch loss
        instance_loss_fn = Discriminative_Loss(0.5, 1.5, 1.0, 1.0, 0.001)
        instance_loss = instance_loss_fn(instance_result, instance_label)
        total_loss = binary_loss + instance_loss
        # we can write a iou in there\
        out = F.softmax(binary_result, dim=1)
        out = torch.argmax(out, dim=1).reshape((batch_size,-1))
        TP = torch.sum(out*binary_label,dim=1) # we get the the N个TP数目
        prediction = torch.sum(out,dim=1)
        label = torch.sum(instance_label.reshape((batch_size,-1)),dim=1)
        iou = torch.mean(TP/(prediction+label-TP))

        return {'binary_result':binary_result,
                'binary_pred':out,
                'instance_result':instance_result,
                'total_loss': total_loss,
                'binary_loss': binary_loss,
                'instance_loss': instance_loss,
                'iou': iou}


# def compute_loss(decoder_result,binary_label,instance_label):
#     '''
#     this func is to compute the loss of lanenet,which is from two branch
#     :param decoder_rsult:
#     :return:
#     '''
#     binary_result,instance_result = decoder_result
#     # the binary branch loss
#     binary_loss = weighted_cross_entropy_loss(binary_result, binary_label)
#     # the instance branch loss
#     instance_loss_fn = Discriminative_Loss(0.5, 1.5, 1.0, 1.0, 0.001)
#     instance_loss = instance_loss_fn(instance_result, instance_label)
#     total_loss = binary_loss + instance_loss
#     # we can write a iou in there
#     out = F.softmax(binary_result, dim=1)
#     out = torch.argmax(out, dim=1)
#     iou = 0
#     batch_size = out.size()[0]
#     for i in range(batch_size):
#         PR = out[i].squeeze(0).nonzero().size()[0]
#         GT = binary_label[i].nonzero().size()[0]
#         TP = (out[i].squeeze(0) * binary_label[i]).nonzero().size()[0]
#         union = PR + GT - TP
#         iou += TP / union
#     iou = iou / batch_size
#
#     return {'total_loss': total_loss, 'binary_loss': binary_loss, 'instance_loss': instance_loss, 'iou': iou}