import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

class Discriminative_Loss(_Loss):

    def __init__(self,delta_var,delta_dist,param_var,param_dist,param_reg):
        super(Discriminative_Loss,self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.param_var = param_var
        self.param_dist = param_dist
        self.param_reg = param_reg

    def forward(self,inputs,targets):
        return self._discriminative_loss(inputs,targets)

    def _discriminative_loss_single(self,input,target):
        # want target is a 2d map
        # to get the background converge，so backgorund is a instance
        # step 1 :get the segmented mu
        feature_dim, height, width = input.size()
        # beaceuse the version,the function hasn't 'return_counted' in pytorch1.0,so we have to change
        # unique_labels, unique_ids, unique_counts = torch.unique(target, sorted=True, return_inverse=True,
        #                                                         return_counts=True)
        unique_labels,unique_ids = torch.unique(target, sorted=True)
        unique_counts = [torch.sum(target==label).item() for label in unique_labels]
        unique_counts = torch.Tensor(unique_counts).cuda()
        # print(unique_labels)
        # print(unique_ids)
        # print(unique_counts)
        instance_num = len(unique_labels)
        # print(instance_num)
        input = input.reshape((height * width, feature_dim))
        target = target.reshape((height, width))
        index = unique_ids.reshape((height * width, 1)).repeat(1, feature_dim)
        # print(input)
        # print(index)
        segmented_sum = torch.zeros(instance_num, feature_dim).cuda().scatter_add(0, index, input)
        mu = torch.div(segmented_sum, unique_counts.reshape((instance_num, 1)))
        # print(segmented_sum)

        # step 2 : calculate the l_var
        # segmented'size instance_num * feature_dim ,counts's size :1* insatnce_num
        mu_expand = torch.gather(mu, 0, index)
        distant = torch.norm(mu_expand - input, dim=1, keepdim=True)
        distant = torch.clamp(distant - self.delta_var, min=0.)
        distant = torch.pow(distant,2)  # the size is (height*width,1)

        l_var = torch.zeros(instance_num, 1).cuda().scatter_add(0, unique_ids.reshape((height * width, 1)), distant)
        l_var = torch.div(l_var, unique_counts.reshape(instance_num, 1))
        l_var = torch.mean(l_var) * self.param_var
        # print(l_var)

        # step 3 :calculate the l_dist
        mu_dim0_expand = mu.repeat(instance_num, 1)
        mu_dim1_expand = mu.repeat(1, instance_num).reshape([instance_num*instance_num,feature_dim])
        # mu_dim1_expand = mu_dim1_expand.reshape((instance_num*instance_num,feature_dim))
        mu_diff = mu_dim1_expand - mu_dim0_expand
        # print(mu_diff)
        # 这里有一个细节的是不需要跟自身的比较
        intermediate_tensor = torch.sum(mu_diff, dim=1, keepdim=True)  # the shape is num_instance * num_instance
        # print(intermediate_tensor)
        bool_mask = (intermediate_tensor != 0).repeat(1, feature_dim)
        mu_diff_need = torch.masked_select(mu_diff, bool_mask).reshape((-1, feature_dim))  # get the 1D tensor

        mu_norm = torch.norm(mu_diff_need, dim=1)
        mu_norm = torch.clamp(2 * self.delta_dist - mu_norm, min=0)
        mu_norm = torch.pow(mu_norm,2)
        l_dist = torch.mean(mu_norm) *self.param_dist
        # step 3: calculate the l_reg
        l_reg = torch.mean(torch.norm(mu, dim=1)) *self.param_reg
        # calculate loss
        loss = l_var + l_dist+ l_reg
        return loss,l_var, l_dist, l_reg

    def _discriminative_loss(self,inputs,targets):
        batch_size = inputs.size(0)
        var_loss = torch.tensor(0,dtype =inputs.dtype,device = inputs.device)
        dist_loss = torch.tensor(0, dtype=inputs.dtype, device=inputs.device)
        reg_loss = torch.tensor(0,dtype =inputs.dtype,device = inputs.device )
        for i in range(batch_size):
            _,l_var,l_dist,l_reg = self._discriminative_loss_single(inputs[i],targets[i])
            var_loss = var_loss + l_var
            dist_loss = dist_loss + l_dist
            reg_loss = reg_loss + l_reg
        var_loss = var_loss / batch_size
        dist_loss = dist_loss /batch_size
        reg_loss = reg_loss/batch_size
        loss = var_loss+dist_loss+reg_loss
        return loss


def weighted_cross_entropy_loss(binary_result,binary_label):
    ''''
    cross_entropy include the softmax ,so we don't need to do it before
    binary_label don't need to be one_hot,it's just a class map
    '''
    # calculate the weights
    size = torch.LongTensor(list(binary_label.size()))
    binary_label_plain = torch.reshape(binary_label,[torch.prod(size).item()])
    unique_class = torch.unique(binary_label_plain)
    print(unique_class)
    counts = [torch.sum(binary_label_plain == label).item() for label in unique_class]
    counts = torch.Tensor(counts).cuda()
    print(counts)
    # if use gou,the weight has to be cuda
    weight = 1.0/torch.log(torch.div(counts,counts.sum().float())+1.02)
    print("the weight of binary label:")
    print(weight)
    ce = nn.CrossEntropyLoss(weight=weight)
    loss = ce(binary_result,binary_label)
    return loss



# # test the loss_single
# instance_num = 4
# height = 3
# width = 4
# input = torch.randn(5,height,width)
# target = torch.LongTensor(height,width).random_() % instance_num
# loss= Discriminative_Loss(0.5, 1.5, 1.0, 1.0, 0.001)
# l_var,l_dist,l_reg  = loss._discriminative_loss_single(input,target)
# print(l_var,l_dist,l_reg)
# def discrimination_loss(instance_result,instance_label,delta_v,delta_d,param_var,param_dist,param_reg):
#     '''
#
#     :param instance_result: NCHW
#     :param instance_label: NHW
#     :param delta_v:
#     :param delta_d:
#     :param param_var:
#     :param param_dist:
#     :param param_reg:
#     :return:
#     '''
#     ## one_step 像素分类
#     batch_size,feature_dim,height,width = instance_result.size()
#
#     size = instance_label.size()
#     instance_label = torch.reshape(instance_label,height,width)
#     unique_class = torch.unique(instance_label)
#     num_instance = unique_class.size()[0]


# classnum=2
# batch_size = 5
# width = 3
# height = 3
# label = torch.LongTensor(batch_size,width,height).random_() % classnum
# result = torch.FloatTensor(batch_size,classnum,width,height).random_()
# loss = weighted_cross_entropy_loss(result,label)
# count = label.sum()
# print(count)
# print(label)
# onehot_label = F.one_hot(label,num_classes=2)
# onehot_label = onehot_label.permute(0,3,1,2)
# print(onehot_label.shape)
# out,count2 = weighted_cross_entropy_loss(onehot_label)
# print(out)
# print(count2)
# onehot_label = torch.zeros(batch_size,classnum,width,height).scatter_(1,label,1)
# print(onehot_label)
#
# # this is the test of loss
# inputs = torch.FloatTensor([0,1,0,0,0,1]).view((2,3))
# outputs= torch.LongTensor([1,2])
# ce = nn.CrossEntropyLoss()
# loss = ce(inputs,outputs)
# print(loss)
# inputs = torch.FloatTensor([0,1,0,0,1,0]).reshape((2,3))
# outputs= torch.LongTensor([1,1])
# ce = nn.CrossEntropyLoss()
# loss = ce(inputs,outputs)
# print(loss)
