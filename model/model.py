from __future__ import division
from model.encoder2 import *
from model.decoder2 import *
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
        encoder_result = self.encoder(forward_input)
        binary_result,instance_result = self.decoder(*encoder_result)
        # after get the result ,we have to compute the loss
        binary_loss = weighted_cross_entropy_loss(binary_result, binary_label)
        # the instance branch loss
        instance_loss_fn = Discriminative_Loss(0.5, 1.5, 1.0, 1.0, 0.001)
        instance_loss = instance_loss_fn(instance_result, instance_label)
        total_loss = binary_loss + instance_loss
        # we can write a iou in there\
        out = F.softmax(binary_result, dim=1)
        binary_pred = torch.argmax(out, dim=1)
        binary_pred_plain = binary_pred.reshape((batch_size,-1))
        TP = torch.sum(binary_pred_plain*binary_label.reshape((batch_size,-1)),dim=1) # we get the the N个TP数目
        prediction = torch.sum(binary_pred_plain,dim=1)
        label = torch.sum(binary_label.reshape((batch_size,-1)),dim=1)
        iou = torch.sum(TP/(prediction+label-TP))/batch_size

        return {'binary_result':binary_result,
                'binary_pred':binary_pred,
                'instance_result':instance_result,
                'total_loss': total_loss,
                'binary_loss': binary_loss,
                'instance_loss': instance_loss,
                'iou': iou,
                'prediction_num':torch.sum(prediction),
                'TP_num':torch.sum(TP)
                }
