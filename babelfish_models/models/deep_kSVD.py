from __future__ import print_function, division
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from ..volume import Vol2D
from ..resnet import ResNet, BasicBlock
from ..super_res import SuperResBlock
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm



class Deep_KSVD(Vol2D):
    def __init__(self, nZ=11, H=232, W=512, nEmbedding=20, prev_frames=1,
                 pred_hidden=20, tensor=T.cuda.FloatTensor):
        super(Deep_KSVD, self).__init__(tensor)
        self.tensor = tensor
        self.nZ = nZ
        self.H = H
        self.W = W
        self.lowH = 8
        self.lowW = 16
        self.lowFeatures = 1
        self.prev_frames = prev_frames
        # batch x channel x Z x H x W
        # Encoding
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], prev_frames)
        self.resOut = 64
        self.nEmbedding = nEmbedding

        # b x 11 x 32 x 11 x 25
        self.embed1 = nn.Linear(self.resOut*self.nZ, nEmbedding)
        self.embed2 = nn.Linear(nEmbedding, nEmbedding)

        # Prediction
        self.pred1 = nn.Linear(nEmbedding+prev_frames, pred_hidden) # add dim for shock_{t+1}
        self.pred_bn1 = nn.BatchNorm1d(pred_hidden)
        self.pred2 = nn.Linear(pred_hidden, nEmbedding) # last 10 (context) are unused

        # Decoding
        self.activation = nn.Tanh()
        self.decoding = nn.Linear(nEmbedding,self.lowFeatures*nZ*self.lowH*self.lowW)
        self.upconv1 = SuperResBlock(2,1,tensor)
        # 11 x 16 x 32
        self.upconv2 = SuperResBlock(2,1,tensor)
        # 11 x 32 x 64
        self.upconv3 = SuperResBlock(2,1,tensor)
        # 11 x 64 x 128
        self.upconv4 = SuperResBlock(2,1,tensor)
        # 11 x 128 x 256
#         self.upconv5 = SuperResBlock(2,tensor)
        # 11 x 256 x 512

        self.tail_decoding = nn.Linear(1,1)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.embed1.weight)

    def sample_embedding(self, mu, logvar):
        if self.training:
            std = T.exp(0.5*logvar)
            eps = T.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, x):
        x = x.transpose(1,2)
        out = self.tensor(x.shape[0],x.shape[1],self.resOut)
        for z in range(x.shape[1]):
            out[:,z], layer_out = self.resnet(x[:,z])
        embedding = self.embed1(out.reshape(x.shape[0],-1))
        embedding = self.embed2(embedding)
        return embedding

    def decode(self, x):
        tail = F.sigmoid(self.tail_decoding(x[:,[0]])) # use first embedding only
        # b x 10
        x = self.activation(self.decoding(x))
        x = x.reshape(x.shape[0],self.nZ,self.lowFeatures,self.lowH,self.lowW)
#         print("upconv1", x.shape)
        x = self.upconv1(x)
#         print("upconv2", x.shape)
        x = self.upconv2(x)
#         print("upconv3", x.shape)
        x = self.upconv3(x)
#         print("upconv4", x.shape)
        x = self.upconv4(x)
#         x = self.upconv5(x)
        x = self.crop(x[:,:,0])
        # squeeze channel
        return x, tail

    def forward(self, x, shock):
        "Return Previous volume (denoised), next volume (prediction), latent mean and logvar."
        encoded = self.encode(x)
        pred = self.decode(encoded)
        return pred, encoded

def train(model,train_data,valid_data, nepochs=10, lr=1e-3, sparse_lambda=1, tail_lambda=1e2, half=False, cuda=True, batch_size=16, num_workers=8):
    global e
    global avg_Y_loss
    global avg_Y_val_loss
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if half:
        optimizer = apex.fp16_utils.FP16_Optimizer(T.optim.Adam(model.parameters(),lr=lr))
    else:
        optimizer = T.optim.Adam(model.parameters(),lr=lr)

    for e in range(nepochs):
        print("epoch {}: ".format(e), end="")
        cum_loss = 0
        cum_Y_loss = 0
        cum_tail_loss = 0
        for batch_data in tqdm(dataloader):
            X, Y = batch_data
            X, X_shock, X_tail = (X["brain"], X["shock"], X["tail_movement"])
            Y, Y_shock, Y_tail = (Y["brain"], Y["shock"], Y["tail_movement"])
            if cuda:
                X = X.cuda()
                Y = Y.cuda()
                X_shock = X_shock.cuda()
                Y_shock = Y_shock.cuda()
                X_tail = X_tail.cuda()
                Y_tail = Y_tail.cuda()
            (Y_pred, Y_pred_tail), embedding = model(X, Y_shock)
            if half:
                X_pred = X_pred.float()
                Y_pred = Y_pred.float()
                embedding = embedding.float()
                logvar = logvar.float()
            mse_Y = F.mse_loss(Y_pred, Y[:,-1])
            mse_tail = F.mse_loss(Y_pred_tail, Y_tail[:,[-1]])
            loss = mse_Y + tail_lambda*mse_tail + sparse_lambda * embedding.norm(1)
            if e==0:
                print("MSE_Y: {:.3E}, Tail: {:.3E}".format(float(mse_Y),float(mse_tail)))
            optimizer.zero_grad()
            if half:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            cum_loss += float(loss)
            cum_Y_loss += float(mse_Y)
            cum_tail_loss += float(mse_tail)

        avg_Y_loss = cum_Y_loss/len(train_data)
        print("avg_loss: {:3E}, Y_loss: {:3E}, tail_loss: {:3E}".format(
            cum_loss/len(train_data), avg_Y_loss, cum_tail_loss/len(train_data)))
        cum_loss = 0
        cum_Y_loss = 0
        cum_tail_loss = 0
        model.eval()
        for batch_data in valid_dataloader:
            X, Y = batch_data
            X, X_shock, X_tail = (X["brain"], X["shock"], X["tail_movement"])
            Y, Y_shock, Y_tail = (Y["brain"], Y["shock"], Y["tail_movement"])
            if cuda:
                X = X.cuda()
                Y = Y.cuda()
                X_shock = X_shock.cuda()
                Y_shock = Y_shock.cuda()
                X_tail = X_tail.cuda()
                Y_tail = Y_tail.cuda()
            (Y_pred, Y_pred_tail), embedding = model(X, Y_shock)
            if half:
                Y_pred = Y_pred.float()
                mean = mean.float()
                logvar = logvar.float()
            mse_Y = F.mse_loss(Y_pred, Y[:,-1])
            mse_tail = F.mse_loss(Y_pred_tail, Y_tail[:,[-1]])
            loss = mse_Y + tail_lambda*mse_tail + sparse_lambda * embedding.norm(1)
            cum_loss += float(loss)
            cum_Y_loss += float(mse_Y)
            cum_tail_loss += float(mse_tail)
        model.train()
        avg_Y_valid_loss = cum_Y_loss/len(valid_data)
        print("VALIDATION: avg_loss: {:3E}, Y_loss: {:3E}, tail_loss: {:3E}".format(
            cum_loss/len(valid_data), avg_Y_valid_loss, cum_tail_loss/len(valid_data)))
    return avg_Y_loss, avg_Y_valid_loss
