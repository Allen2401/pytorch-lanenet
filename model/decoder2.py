from model.blocks2 import *


class EnetDecoder(nn.Module):
    # include bottleneck3 bottleneck4 and bottleneck5

    def __init__(self,instance_depth,encoder_relu=False,decoder_relu=True):
        '''
        the instance branch of decoder
        '''
        super(EnetDecoder, self).__init__()
        # binary_branch
        # Stage 3 - Encoder
        self.binary_regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.binary_dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.binary_asymmetric3_2 = RegularBottleneck(128,kernel_size=5,padding=2,asymmetric=True,dropout_prob=0.1,relu=encoder_relu)
        self.binary_dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.binary_regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.binary_dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.binary_asymmetric3_6 = RegularBottleneck(128,kernel_size=5,asymmetric=True,padding=2,dropout_prob=0.1,relu=encoder_relu)
        self.binary_dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 4 - Decoder
        self.binary_upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.binary_regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.binary_regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.binary_upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.binary_regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.binary_transposed_conv = nn.ConvTranspose2d(16,2,kernel_size=3,stride=2,padding=1,bias=False)


        ## instance branch
        self.instance_regular3_0 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.instance_dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.instance_asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.instance_dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.instance_regular3_4 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.instance_dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.instance_asymmetric3_6 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.instance_dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 4 - Decoder
        self.instance_upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.instance_regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.instance_regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.instance_upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.instance_regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.instance_transposed_conv = nn.ConvTranspose2d(16, instance_depth, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self,input,input_size,indices,output_size):
        '''

        :param encoder_input: the result of encoder part
        :param indices: the pooling indices in encoder
        :return:
        --binary:the result of binary branch
        --insatcne:the result of instance branch
        '''
        # binary_branch
        # Stage 3 - Encoder
        binary = self.binary_regular3_0(input)
        binary = self.binary_dilated3_1(binary)
        binary = self.binary_asymmetric3_2(binary)
        binary = self.binary_dilated3_3(binary)
        binary = self.binary_regular3_4(binary)
        binary = self.binary_dilated3_5(binary)
        binary = self.binary_asymmetric3_6(binary)
        binary = self.binary_dilated3_7(binary)

        # Stage 4 - Decoder
        binary = self.binary_upsample4_0(binary, indices[1], output_size=output_size[1])
        binary = self.binary_regular4_1(binary)
        binary = self.binary_regular4_2(binary)

        # Stage 5 - Decoder
        binary = self.binary_upsample5_0(binary, indices[0], output_size=output_size[0])
        binary = self.binary_regular5_1(binary)
        binary = self.binary_transposed_conv(binary, output_size=input_size)

        # instance branch
        # Stage 3 - Encoder
        instance = self.instance_regular3_0(input)
        instance = self.instance_dilated3_1(instance)
        instance = self.instance_asymmetric3_2(instance)
        instance = self.instance_dilated3_3(instance)
        instance = self.instance_regular3_4(instance)
        instance = self.instance_dilated3_5(instance)
        instance = self.instance_asymmetric3_6(instance)
        instance = self.instance_dilated3_7(instance)

        # Stage 4 - Decoder
        instance = self.instance_upsample4_0(instance, indices[1], output_size=output_size[1])
        instance = self.instance_regular4_1(instance)
        instance = self.instance_regular4_2(instance)

        # Stage 5 - Decoder
        instance = self.instance_upsample5_0(instance, indices[0], output_size=output_size[0])
        instance = self.instance_regular5_1(instance)
        instance = self.instance_transposed_conv(instance, output_size=input_size)
        return binary,instance
