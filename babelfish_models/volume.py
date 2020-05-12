import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Vol2D(nn.Module):
    "Use same 2D operations mapped over each z slice"
    def __init__(self, tensor=T.cuda.FloatTensor):
        super(Vol2D, self).__init__()
        self.tensor = tensor

    def vol_PixelShuffle(self, x):
        # Helper for subpixel convolution
        first = self.pixel_shuffle(x[:,0])
        # b x z x C x H x W
        ret = self.tensor(x.shape[0],x.shape[1],first.shape[1], first.shape[2], first.shape[3])
        ret[:,0] = first
        for z in range(1,x.shape[1]):
            ret[:,z] = self.pixel_shuffle(x[:,z])
        return ret

        # batch x Z*C x H x W
        input = x.view(x.shape[0],-1,x.shape[3],x.shape[4])
        pooled = F.max_pool2d(input,kernel_size)
        return pooled.reshape(pooled.shape[0],x.shape[1],x.shape[2],pooled.shape[2],pooled.shape[3])

    def vol_MaxPool2d(self, x, kernel_size):
        # batch x Z*C x H x W
        input = x.view(x.shape[0],-1,x.shape[3],x.shape[4])
        pooled = F.max_pool2d(input,kernel_size)
        return pooled.reshape(pooled.shape[0],x.shape[1],x.shape[2],pooled.shape[2],pooled.shape[3])

    def vol_BatchNorm2d(self, x, bn):
        activations = self.tensor(x.shape)
        for z in range(x.shape[1]):
            activations[:,z] = bn(x[:,z].contiguous())
        return activations

    def vol_conv2d(self, x, weight, pad):
        # batch x Z x C x H x W
        activations = self.tensor(x.shape[0],x.shape[1],weight.shape[0],x.shape[3],x.shape[4])
        for z in range(x.shape[1]):
            activations[:,z] = F.conv2d(x[:,z], weight, padding=pad)
        return activations

    def crop(self, x):
        cropH = (x.shape[2] - self.H)/2
        cropW = (x.shape[3] - self.W)/2
        if cropH>0:
            x = x[:,:,int(np.floor(cropH)):-int(np.ceil(cropH))]
        if cropW>0:
            x = x[:,:,:,int(np.floor(cropW)):-int(np.ceil(cropW))]
        return x

def volume_mse(X, Y):
    with T.no_grad():
        loss = F.mse_loss(X,Y,reduce=False).reshape(X.shape[0],-1).sum(1)
    return loss
