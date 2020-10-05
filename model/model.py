from __future__ import division
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
        if len(input)==3:
            forward_input = input[0]
            binary_label = input[1]
            instance_label = input[2]
        else:
            forward_input = input[0]
        batch_size = forward_input.size(0)
        # encoder_result,indices = self.encoder(forward_input)
        # binary_result,instance_result = self.decoder(encoder_result,indices)
        encoder_result = self.encoder(forward_input)
        binary_result,instance_result = self.decoder(*encoder_result)
        binary_out = F.softmax(binary_result, dim=1)
        # shape :N*h*w
        binary_out = torch.argmax(binary_out, dim=1)
        if len(input)!=3:
            return {
                   'binary_pred': binary_out,
                   'instance_result': instance_result
            }
        # after get the result ,we have to compute the loss
        # binary_loss = weighted_cross_entropy_loss(binary_result, binary_label)
        binary_loss = weighted_labelsmooth_loss(binary_result, binary_label)
        # the instance branch loss
        instance_loss_fn = Discriminative_Loss(0.5, 1.5, 1.0, 1.0, 0.001)
        instance_loss = instance_loss_fn(instance_result, instance_label)
        # reg loss
        reg_loss = torch.tensor(0.0).cuda()
        for name,param in self.encoder.named_parameters():
            if 'weight' in name:
                l2_reg = torch.norm(param,p=2)
                reg_loss = reg_loss + l2_reg
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                l2_reg = torch.norm(param, p=2)
                reg_loss = reg_loss + l2_reg
        total_loss = binary_loss + 1.5 * instance_loss + 0.001 * reg_loss
         #+ 0.0005 * reg_loss
        # we can write a iou in there
        binary_out_plain = binary_out.reshape((batch_size,-1))
        TP = torch.sum(binary_out_plain*binary_label.reshape((batch_size,-1)),dim=1) # we get the the N个TP数目
        prediction = torch.sum(binary_out_plain,dim=1)
        label = torch.sum(binary_label.reshape((batch_size,-1)),dim=1)
        iou = torch.sum(TP * 1.0/(prediction+label-TP))/batch_size
        return {'binary_result':binary_result,
                'binary_pred':binary_out,
                'instance_result':instance_result,
                'total_loss': total_loss,
                'binary_loss': binary_loss,
                'instance_loss': instance_loss,
                'iou': iou,
                'prediction_num':torch.mean(prediction*1.),
                'TP_num':torch.mean(TP*1.)
        }


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