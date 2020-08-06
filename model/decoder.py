from model.blocks import *


class EnetDecoder(nn.Module):
    # include bottleneck3 bottleneck4 and bottleneck5

    def __init__(self,instance_depth):
        '''
        the insatnce branch of decoder
        '''
        super(EnetDecoder, self).__init__()
        # bottleneck3
        self.bottleneck_instance3 = nn.Sequential(
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
        self.bottleneck_instance4_0 = Upsamplebottleneck(128, 64, 64, 2, 2)
        self.bottleneck_instacne4_1 = nn.Sequential(
            RegularBottleNeck(64, 64, 16, 3, padding=1),
            RegularBottleNeck(64, 64, 16, 3, padding=1)
        )
        # bottleneck5
        self.bottleneck_instance5_0 = Upsamplebottleneck(64, 16, 16, 2, 2)
        self.bottleneck_instance5_1 = RegularBottleNeck(16, 16, 4, 3, padding=1)
        # fullconv
        self.fullconv_instance = nn.ConvTranspose2d(16, instance_depth, 2, 2, bias=False)

        '''
        the binary branch of decoder
        '''
        # bottleneck3
        self.bottleneck_binary3 = nn.Sequential(
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
        self.bottleneck_binary4_0 = Upsamplebottleneck(128, 64, 64, 2, 2)
        self.bottleneck_binary4_1 = nn.Sequential(
            RegularBottleNeck(64, 64, 16, 3, padding=1),
            RegularBottleNeck(64, 64, 16, 3, padding=1)
        )
        # bottleneck5
        self.bottleneck_binary5_0 = Upsamplebottleneck(64, 16, 16, 2, 2)
        self.bottleneck_binary5_1 = RegularBottleNeck(16, 16, 4, 3, padding=1)
        # fullconv
        self.fulconv_binary = nn.ConvTranspose2d(16, 2, 2, 2, bias=False)

    def forward(self,encoder_result,indices):
        '''

        :param encoder_input: the result of encoder part
        :param indices: the pooling indices in encoder
        :return:
        --binary:the result of binary branch
        --insatcne:the result of instance branch
        '''

        # instance branch
        instance = self.bottleneck_instance3(encoder_result)
        instance =  self.bottleneck_instance4_0(instance,indices[1])
        instance = self.bottleneck_instacne4_1(instance)
        instance = self.bottleneck_binary5_0(instance,indices[0])
        instance = self.fullconv_instance(instance)

        # branch branch
        binary = self.bottleneck_binary3(encoder_result)
        binary = self.bottleneck_binary4_0(binary,indices[1])
        binary = self.bottleneck_instacne4_1(binary)
        binary = self.bottleneck_binary5_0(binary,indices[0])
        binary  =self.bottleneck_instance5_1(binary)
        binary = self.fulconv_binary(binary)
        return binary,instance
