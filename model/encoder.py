from model.blocks import *
import torch.nn as nn


# this file is the encoder of lanenet,the baseline is enet
class EnetEncoder(nn.Module):

    def __init__(self):

        # include three stage:initial  bottleneck1 and bottleneck2
        super(EnetEncoder,self).__init__()
        self.InitialBlock = InitialBlock(13, 0)
        # bottleneck1
        # the downsamping must be alone
        self.bottleneck1_0 = RegularBottleNeck(16, 64, 16, 2, 2, downsample=True)
        self.bottleneck1_1 = nn.Sequential(
            RegularBottleNeck(64, 64, 16, 3, padding=1),
            RegularBottleNeck(64, 64, 16, 3, padding=1),
            RegularBottleNeck(64, 64, 16, 3, padding=1),
            RegularBottleNeck(64, 64, 16, 3, padding=1)
        )
        ## bottleneck2
        self.bottleneck2_0 = RegularBottleNeck(64, 128, 64, 2, 2, downsample=True)
        self.bottleneck2_1 = nn.Sequential(
            RegularBottleNeck(128, 128, 64, 3, padding=1),
            RegularBottleNeck(128, 128, 64, 3, dilation=2, padding=2),
            RegularBottleNeck(128, 128, 64, 5, padding=2, asymmetric=True),
            RegularBottleNeck(128, 128, 64, 3, dilation=4, padding=4),
            RegularBottleNeck(128, 128, 64, 3, padding=1),
            RegularBottleNeck(128, 128, 64, 3, dilation=8, padding=8),
            RegularBottleNeck(128, 128, 64, 5, padding=2, asymmetric=True),
            RegularBottleNeck(128, 128, 64, 3, dilation=16, padding=16)
        )
    def forward(self,input):
        x = self.InitialBlock(input)
        # bottleneck1
        x, indices1 = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        # bottleneck2
        x, indices2 = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        return x,[indices1,indices2]

