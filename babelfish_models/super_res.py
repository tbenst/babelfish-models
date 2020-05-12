import torch as T
import torch.nn as nn
from .volume import Vol2D

class SuperResBlock(Vol2D):
    """Upsample Volume using subpixel convolution.

    Reference: https://arxiv.org/pdf/1609.05158.pdf"""
    def __init__(self, upscale_factor, in_channels=1, tensor=T.cuda.FloatTensor, out_channels=1):
        super(SuperResBlock, self).__init__(tensor)
        self.tensor = tensor
        self.activation = nn.ReLU()
        self.dconv1 = nn.Parameter(self.tensor(64,in_channels,5,5))
        self.dpad1 = (2,2)
        self.dbn1 = nn.BatchNorm2d(64)
        self.dconv2 = nn.Parameter(self.tensor(64,64,3,3))
        self.dpad2 = (1,1)
        self.dbn2 = nn.BatchNorm2d(64)
        self.dconv3 = nn.Parameter(self.tensor(32,64,3,3))
        self.dpad3 = (1,1)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dconv4 = nn.Parameter(self.tensor(out_channels*upscale_factor**2,32,3,3))
        self.dpad4 = (1,1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.initialize_weights()

    def forward(self, x):
        x = self.activation(self.vol_BatchNorm2d(self.vol_conv2d(x, self.dconv1, self.dpad1), self.dbn1))
        x = self.activation(self.vol_BatchNorm2d(self.vol_conv2d(x, self.dconv2, self.dpad2), self.dbn2))
        x = self.activation(self.vol_BatchNorm2d(self.vol_conv2d(x, self.dconv3, self.dpad3), self.dbn3))
        x = self.vol_conv2d(x, self.dconv4, self.dpad4)
        x = self.vol_PixelShuffle(x)
        return x

    def initialize_weights(self):
        if self.tensor==T.cuda.FloatTensor:
            nn.init.orthogonal_(self.dconv1, nn.init.calculate_gain('relu'))
            nn.init.orthogonal_(self.dconv2, nn.init.calculate_gain('relu'))
            nn.init.orthogonal_(self.dconv3, nn.init.calculate_gain('relu'))
            nn.init.orthogonal_(self.dconv4)
        else:
            for m in [self.dconv1, self.dconv2, self.dconv3, self.dconv4]:
                nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')
        for bn in [self.dbn1,self.dbn2,self.dbn3]:
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)

class SuperResSkip(Vol2D):
    """Upsample Volume using subpixel convolution.

    Reference: https://arxiv.org/pdf/1609.05158.pdf"""
    def __init__(self, upscale_factor, in_channels=1, tensor=T.cuda.FloatTensor, out_channels=1):
        super(SuperResSkip, self).__init__(tensor)
        self.tensor = tensor
        self.activation = nn.ReLU()
        self.dconv1 = nn.Parameter(self.tensor(64,in_channels,5,5))
        self.dpad1 = (2,2)
        self.dbn1 = nn.BatchNorm2d(64)
        self.dconv2 = nn.Parameter(self.tensor(64,64,3,3))
        self.dpad2 = (1,1)
        self.dbn2 = nn.BatchNorm2d(64)
        self.dconv3 = nn.Parameter(self.tensor(32,64,3,3))
        self.dpad3 = (1,1)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dconv4 = nn.Parameter(self.tensor(out_channels*upscale_factor**2,32,3,3))
        self.dpad4 = (1,1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.initialize_weights()

    def forward(self, x, layer_output):
        # x: B x Z x C x H x W
        x_chan = x.shape[2]
        new_x = self.tensor(x.shape[0], x.shape[1], x_chan+layer_output.shape[2],
                       x.shape[3], x.shape[4])
        new_x[:,:,:x_chan] = x
#         print(new_x.shape, layer_output.shape)
        new_x[:,:,x_chan:] = layer_output
        x = self.activation(self.vol_BatchNorm2d(self.vol_conv2d(new_x, self.dconv1, self.dpad1), self.dbn1))
        del new_x
        x = self.activation(self.vol_BatchNorm2d(self.vol_conv2d(x, self.dconv2, self.dpad2), self.dbn2))
        x = self.activation(self.vol_BatchNorm2d(self.vol_conv2d(x, self.dconv3, self.dpad3), self.dbn3))
        x = self.vol_conv2d(x, self.dconv4, self.dpad4)
        x = self.vol_PixelShuffle(x)
        return x

    def initialize_weights(self):
        if self.tensor==T.cuda.FloatTensor:
            nn.init.orthogonal_(self.dconv1, nn.init.calculate_gain('relu'))
            nn.init.orthogonal_(self.dconv2, nn.init.calculate_gain('relu'))
            nn.init.orthogonal_(self.dconv3, nn.init.calculate_gain('relu'))
            nn.init.orthogonal_(self.dconv4)
        else:
            for m in [self.dconv1, self.dconv2, self.dconv3, self.dconv4]:
                nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')
        for bn in [self.dbn1,self.dbn2,self.dbn3]:
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
