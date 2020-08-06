import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


class InitialBlock(nn.Module):
    '''two branch:
    1.conv bracnch :generally,3*3 kernel and 2 stride
    2.maxpooling branch:non-overlapping maxpooling

    inupt:
    default input channel num is 3,kernel size is 3*3 ,and the stride is 2.
    --out_channel:the output channel num of conv branch
    --padding: because the input_size is unknown,so need this param,the default is 0
    --batch_normalization: default:true.
    '''

    def __init__(self, out_channel, padding=1):

        super(InitialBlock,self).__init__()
        self.conv  = nn.Conv2d(3,out_channel,kernel_size=3,stride=2,padding=1,bias=False) # without the padding ,the output size is 255
        self.maxpool = nn.MaxPool2d(2,2)
        self.batchnorm = nn.BatchNorm2d(out_channel+3)
        self.relu = nn.PReLU()

    def forward(self,input):
        '''
        :param input:the input data.the default channel size is 3
        :return:
        '''
        input_conv = self.conv(input)
        input_pool= self.maxpool(input)
        x = torch.cat([input_conv,input_pool],dim = 1)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class Upsamplebottleneck(nn.Module):
    '''
    two branch
    main branch:the unpooling layer , expansion of channel
    extension branch:1*1,deconvolution,1*1 and dropout
    the param:
    --input_channel
    --output_channel
    --inter_channel
    --kernel_size:the kernel_size of deconv,the stride is dedault to 2
    --padding :the padding of deconv
    --pdrop
    '''
    def __init__(self, input_channel, output_channel, internal_channel, kernel_size, stride, padding=0,pdrop=0):
        # the conv is before unpool
        super(Upsamplebottleneck,self).__init__()
        self.main_conv = nn.Sequential(nn.Conv2d(input_channel, output_channel, 1, bias=False),
                                       nn.BatchNorm2d(output_channel))
        self.unpool = nn.MaxUnpool2d(2)  # the stride is set to kernel_szie by the default
        self.extension = nn.Sequential(
            nn.Conv2d(input_channel,internal_channel,1,bias=False),
            nn.BatchNorm2d(internal_channel),
            nn.PReLU(),
            nn.ConvTranspose2d(internal_channel,internal_channel,kernel_size=kernel_size,stride=stride,padding = padding,bias=False),
            nn.BatchNorm2d(internal_channel),
            nn.PReLU(),
            nn.Conv2d(internal_channel,output_channel,1,bias=False),
            nn.BatchNorm2d(output_channel),
            nn.Dropout2d(pdrop)
        )
        self.relu = nn.PReLU()

    def forward(self,input,max_indices):
        # the max_indices is for unpooling
        main = self.main_conv(input)
        main = self.unpool(main,max_indices)
        extension = self.extension(input)
        output=  self.relu(main + extension)
        return output


class RegularBottleNeck(nn.Module):
    def __init__(self, input_channel, output_channel, internal_channel, kernel_size,stride=1,padding=0, downsample=False,
                 dilation=1, asymmetric=False, pdrop=0):
        '''
        in regular mode,the main branch is shortcut,but in downsampling,the main branch needs conv
        the extension branch's conv includes conv,dilated and asymettric
        '''
        super(RegularBottleNeck,self).__init__()
        self.downsample=downsample
        if downsample:
            self.maxpool = nn.MaxPool2d(2, 2,return_indices=True) # to get the index of the max
            self.main_conv = nn.Sequential(nn.Conv2d(input_channel,output_channel,1,bias=False)
                                           ,nn.BatchNorm2d(output_channel))

        self.extension_conv1=nn.Sequential(nn.Conv2d(input_channel,internal_channel,1,bias = False),
                                           nn.BatchNorm2d(internal_channel),
                                           nn.PReLU())
        if asymmetric:
            self.extension_conv2 = nn.Sequential(nn.Conv2d(internal_channel,internal_channel,kernel_size=(kernel_size,1),padding=(padding,0),bias = False),
                                                 nn.Conv2d(internal_channel,internal_channel,kernel_size=(1,kernel_size),padding=(0,padding),bias = False),
                                                 nn.BatchNorm2d(internal_channel),
                                                 nn.PReLU())
        else:
            # if downsample:
            #     self.extension_conv2 = nn.Sequential(nn.Conv2d(internal_channel,internal_channel,2,2,padding= padding,bias=False),
            #                                          nn.BatchNorm2d(internal_channel),
            #                                         nn.PReLU())
            # else:
                # when the kernel_size of dilated is set to 3,the padding is equal to dilated
            self.extension_conv2 = nn.Sequential(nn.Conv2d(internal_channel,internal_channel,kernel_size,stride,padding=padding,dilation=dilation,bias=False),
                                                 nn.BatchNorm2d(internal_channel),
                                                 nn.PReLU())
        self.extension_conv3 = nn.Sequential(nn.Conv2d(internal_channel,output_channel,1,bias=False),
                                             nn.BatchNorm2d(output_channel),nn.Dropout2d(pdrop))
        self.relu = nn.PReLU()
    def forward(self,input):
        if self.downsample:
            max,indices = self.maxpool(input)
            main = self.main_conv(max)
        else:
            main = input
        extension = self.extension_conv1(input)
        extension = self.extension_conv2(extension)
        extension = self.extension_conv3(extension)

        output = self.relu(main+extension)
        if self.downsample:
            return output,indices
        return output


class ENet(nn.Module):
    '''
    the class is the arch of the network ENet
    '''
    def __init__(self,class_num):

        super(ENet,self).__init__()
        self.InitialBlock = InitialBlock(13,0)
        #bottleneck1
        #the downsamping must be alone
        self.bottleneck1_0 =  RegularBottleNeck(16,64,16,2,2,downsample=True)
        self.bottleneck1_1 = nn.Sequential(
            RegularBottleNeck(64, 64, 16, 3, padding=1),
            RegularBottleNeck(64, 64, 16, 3, padding=1),
            RegularBottleNeck(64, 64, 16, 3, padding=1),
            RegularBottleNeck(64, 64, 16, 3, padding=1)
        )
        ## bottleneck2
        self.bottleneck2_0 = RegularBottleNeck(64,128,64,2,2,downsample=True)
        self.bottleneck2_1 = nn.Sequential(
            RegularBottleNeck(128,128,64,3,padding=1),
            RegularBottleNeck(128,128,64,3,dilation=2,padding=2),
            RegularBottleNeck(128,128,64,5,padding=2,asymmetric=True),
            RegularBottleNeck(128,128,64,3,dilation=4,padding=4),
            RegularBottleNeck(128, 128, 64, 3, padding=1),
            RegularBottleNeck(128,128,64,3,dilation=8,padding=8),
            RegularBottleNeck(128,128,64,5,padding=2,asymmetric=True),
            RegularBottleNeck(128,128,64,3,dilation=16,padding=16)
        )
        # bottleneck3
        self.bottleneck3 = nn.Sequential(
            RegularBottleNeck(128, 128, 64, 3, padding=1),
            RegularBottleNeck(128, 128, 64, 3, dilation=2, padding=2),
            RegularBottleNeck(128, 128, 64, 5, padding=2, asymmetric=True),
            RegularBottleNeck(128, 128, 64, 3, dilation=4, padding=4),
            RegularBottleNeck(128, 128, 64, 3, padding=1),
            RegularBottleNeck(128, 128, 64, 3, dilation=8, padding=8),
            RegularBottleNeck(128, 128, 64, 5, padding=2, asymmetric=True),
            RegularBottleNeck(128, 128, 64, 3, dilation=16, padding=16)
        )
        # bottleneck4
        self.bottleneck4_0 = Upsamplebottleneck(128,64,64,2,2)
        self.bottleneck4_1 = nn.Sequential(
            RegularBottleNeck(64,64,16,3,padding=1),
            RegularBottleNeck(64,64,16,3,padding=1)
        )
        # bottleneck5
        self.bottleneck5_0 = Upsamplebottleneck(64,16,16,2,2)
        self.bottleneck5_1 = RegularBottleNeck(16,16,4,3,padding=1)
        # fullconv
        self.fullconv = nn.ConvTranspose2d(16, class_num, 2, 2, bias=False)

    def forward(self,input):
        x = self.InitialBlock(input)
        # bottleneck1
        x,indices1 = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        # bottleneck2
        x,indices2= self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        # bottleneck3
        x = self.bottleneck3(x)
        # bottleneck4
        x = self.bottleneck4_0(x,indices2)
        x = self.bottleneck4_1(x)
        # bottelneck5
        x = self.bottleneck5_0(x,indices1)
        x = self.bottleneck5_1(x)
        # fullconv
        x = self.fullconv(x)
        return x

# make a tensor batch*3*512*512
if __name__ == '__main__':
    img = torch.randn(5,3,512,512)
    model = ENet(3)
    output = model(img)
    print(output.shape)