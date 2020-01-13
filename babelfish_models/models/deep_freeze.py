from __future__ import print_function, division
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from ..volume import Vol2D
from ..resnet import ResNet, BasicBlock
from ..super_res import SuperResSkip
from torch.utils.data import DataLoader, Dataset
from ..misc import sigmoid_schedule
from tqdm import tqdm
import numpy as np

#
# class Base(Vol2D):
#     "Common functionality for Encoder, Predictor, and Decoder"
#     def __init__(self, tensor=T.cuda.FloatTensor):
#         super(Base, self).__init__()
#         self.tensor = tensor

class Encoder(Vol2D):
    def __init__(self, nZ=11, H=232, W=512, nEmbedding=20, prev_frames=1, next_frames=1,
                 pred_hidden=20, tensor=T.cuda.FloatTensor):
        super(Encoder, self).__init__(tensor)
        self.tensor = tensor
        self.nZ = nZ
        self.H = H
        self.W = W
        self.prev_frames = prev_frames
        # batch x channel x Z x H x W
        # Encoding
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], prev_frames)
        self.resOut = 64
        self.nEmbedding = nEmbedding
        assert nEmbedding % 2 == 0

        # b x 11 x 32 x 11 x 25
        self.encoding_mean = nn.Linear(self.resOut*self.nZ, nEmbedding)
        self.encoding_logvar = nn.Linear(self.resOut*self.nZ, nEmbedding)

        nn.init.xavier_normal_(self.encoding_mean.weight)
        # TODO - make larger?
        nn.init.xavier_normal_(self.encoding_logvar.weight,1e-3)


    def sample_embedding(self, mu, logvar):
        if self.training:
            std = T.exp(0.5*logvar)
            eps = T.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        x = x.transpose(1,2)
        # X :: b x z x t x h x w
        out = self.tensor(x.shape[0],x.shape[1],self.resOut)
        layers = ["conv1_out", "layer1_out", "layer2_out", "layer3_out", "layer4_out"]
        layer_outputs = {k: [] for k in layers}
        for z in range(x.shape[1]):
            out[:,z], layer_out = self.resnet(x[:,z])
            for k in layers:
                layer_outputs[k].append(layer_out[k])
        layer_outputs = {k: T.stack(v,1) for k,v in layer_outputs.items()}
        mean = self.encoding_mean(out.reshape(x.shape[0],-1))
        logvar = self.encoding_logvar(out.reshape(x.shape[0],-1))
        return mean, logvar, layer_outputs


class Predictor(Vol2D):
    def __init__(self, nZ=11, H=232, W=512, nEmbedding=20, prev_frames=1, next_frames=1,
        pred_hidden=20, tensor=T.cuda.FloatTensor):
        super(Predictor, self).__init__(tensor)
        self.activation = nn.Tanh()
        nhalf_embed = int(nEmbedding/2)
        self.pred1 = nn.Linear(nhalf_embed+next_frames, pred_hidden) # add dims for shock_{t+1}
        self.pred2 = nn.Linear(pred_hidden, nhalf_embed) # last 10 (context) are unused

    def forward(self, x, shock):
        "Consumes embedding, and returns predicted embedding."
        x = T.cat([x, shock],1)
        x = self.activation(self.pred1(x))
        x = self.pred2(x)
        return x

class Decoder(Vol2D):
    def __init__(self, nZ=11, H=232, W=512, nEmbedding=20, prev_frames=1, next_frames=1,
        pred_hidden=20, tensor=T.cuda.FloatTensor):

        super(Decoder, self).__init__(tensor)
        # Decoding
        self.activation = nn.Tanh()
        # only use 10 embeddings for frame decoding, the other 10 are context
        self.lowFeatures = 1
        self.lowH = 8
        self.lowW = 16
        self.nEmbedding = nEmbedding
        self.H = H
        self.W = W
        self.nZ = nZ
        self.decoding = nn.Linear(nEmbedding/2,self.lowFeatures*nZ*self.lowH*self.lowW)
        self.upconv1 = SuperResSkip(2,65,tensor)
        # 11 x 16 x 32
        self.upconv2 = SuperResSkip(2,65,tensor)
        # 11 x 32 x 64
        self.upconv3 = SuperResSkip(2,65,tensor)
        # 11 x 64 x 128
        self.upconv4 = SuperResSkip(2,65,tensor)
        # 11 x 128 x 256
#         self.upconv5 = SuperResSkip(2,tensor)
        # 11 x 256 x 512

        self.tail_decoding = nn.Linear(1,1)

    def forward(self, x, layer_output):
        tail = F.sigmoid(self.tail_decoding(x[:,[0]])) # use first embedding only
        # b x 10
        # only use first half for brain data
        x = self.activation(self.decoding(x)) # TODO this line differs from deep_skip 108
        x = x.reshape(x.shape[0],self.nZ,self.lowFeatures,self.lowH,self.lowW)
#         print("upconv1", x.shape)
        x = self.upconv1(x, layer_output["layer3_out"])
#         print("upconv2", x.shape)
        x = self.upconv2(x, layer_output["layer2_out"])
#         print("upconv3", x.shape)
        x = self.upconv3(x, layer_output["layer1_out"])
#         print("upconv4", x.shape)
        x = self.upconv4(x, layer_output["conv1_out"])
#         x = self.upconv5(x)
        x = self.crop(x[:,:,0])
        # squeeze channel
        return x, tail


class DeepFreeze(Vol2D):
    def __init__(self, nZ=11, H=232, W=512, nEmbedding=20, prev_frames=1, next_frames=1,
        pred_hidden=20, tensor=T.cuda.FloatTensor):

        super(DeepFreeze, self).__init__(tensor)

        self.nhalf_embed = int(nEmbedding/2)
        self.encode = Encoder(nZ, H, W, nEmbedding, prev_frames,
            next_frames, pred_hidden, tensor)
        self.predictor = Predictor(nZ, H, W, nEmbedding, prev_frames, next_frames, pred_hidden, tensor)
        self.denoiser = Predictor(nZ, H, W, nEmbedding, prev_frames, next_frames, pred_hidden, tensor)
        self.decode = Decoder(nZ, H, W, nEmbedding, prev_frames,
            next_frames, pred_hidden, tensor)

    def forward(self, x, x_shock=None, y_shock=None, model="both"):
        """Return Previous volume (denoised) and/or next volume (prediction), latent mean and logvar.

        Selector is due to DataParallel restriction"""
        if model=="both":
            mean, logvar, layer_outputs = self.encode(x)
            if True:#self.training:
                encoded = self.encode.sample_embedding(mean, logvar)
            else:
                encoded = mean
            encoded_prev = self.denoiser(encoded[:,:self.nhalf_embed], y_shock) # should be x_shock? TODO
            encoded_pred = self.predictor(encoded[:,self.nhalf_embed:], y_shock)
            prev = self.decode(encoded_prev, layer_outputs)
            pred = self.decode(encoded_pred, layer_outputs)
            return prev, pred, mean, logvar
        elif model=="denoise":
            return self.denoise(x, x_shock)
        elif model=="predict":
            return self.predict(x, y_shock)
        else:
            raise("Bad choice of model: must be denoise or predict or both")

    def denoise(self, x, shock):
        mean, logvar, layer_outputs = self.encode(x)
        if True:#self.training:
            encoded = self.encode.sample_embedding(mean, logvar)
        else:
            encoded = mean
        encoded_prev = self.denoiser(encoded[:,:self.nhalf_embed], shock)
        prev = self.decode(encoded_prev, layer_outputs)
        return prev, mean, logvar

    def predict(self, x, shock):
        mean, logvar, layer_outputs = self.encode(x)
        if True:#self.training:
            encoded = self.encode.sample_embedding(mean, logvar)
        else:
            encoded = mean
        encoded_pred = self.predictor(encoded[:,self.nhalf_embed:], shock)
        pred = self.decode(encoded_pred, layer_outputs)
        return pred, mean, logvar

    def freeze(self):
        "Freezes Encoder + decoder, leaving only predictor trainable."
        for param in self.parameters():
            param.requires_grad = False
        for param in self.predictor.parameters():
            param.requires_grad = True

    def unfreeze(self):
        "Freezes Encoder + decoder, leaving only predictor trainable."
        for param in self.parameters():
            param.requires_grad = True

def unit_norm_KL_divergence(mu, logvar):
    "Reconstruction + KL divergence losses summed over all elements and batch."
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp())

def train_helper(model, dataloader, optimizer, nExamples, X_lambda, Y_lambda, validation=False, half=False, verbose=True, cuda=True, kl_lambda=1, kl_tail=1):
    cum_loss = 0
    cum_X_loss = 0
    cum_Y_loss = 0
    cum_kld_loss = 0
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
        if Y_lambda==1 and X_lambda==0:
            (y_pred, y_pred_tail), mean, logvar = model(X, y_shock=Y_shock, model="predict")
            x_pred = y_pred # will not be used..
            x_pred_tail = y_pred_tail # will not be used..
        elif X_lambda==1 and Y_lambda==0:
            (x_pred, x_pred_tail), mean, logvar = model(X, x_shock=X_shock, model="denoise")
            y_pred = x_pred # will not be used..
            y_pred_tail = x_pred_tail # will not be used..
        elif X_lambda==1 and Y_lambda==1:
            (x_pred, x_pred_tail), (y_pred, y_pred_tail), mean, logvar = model(X, x_shock=X_shock, y_shock=Y_shock, model="both")
        else:
            raise("Y_lambda or X_lambda must be 1")
        if half:
            pred = pred.float()
            mean = mean.float()
            logvar = logvar.float()
        kld = unit_norm_KL_divergence(mean, logvar)
        mse_X = F.mse_loss(x_pred, Y[:,0])
        mse_Y = F.mse_loss(y_pred, Y[:,-1])
        mse_X_tail = F.mse_loss(x_pred_tail, Y_tail[:,[0]])
        mse_Y_tail = F.mse_loss(y_pred_tail, Y_tail[:,[-1]])
        mse_tail = X_lambda * mse_X_tail + Y_lambda * mse_Y_tail
        loss = X_lambda * mse_X + Y_lambda * mse_Y + \
            kl_lambda * kld + kl_tail*mse_tail

        if verbose:
            print("\nMSE_X: {:.3E}, MSE_Y: {:.3E}, KLD: {:.3E}, Tail: {:.3E}".format(float(mse_X),float(mse_Y),float(kld),float(mse_tail)))
        if validation:
            optimizer.zero_grad()
            if half:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()
        cum_loss += float(loss)
        cum_X_loss += float(mse_X)
        cum_Y_loss += float(mse_Y)
        cum_kld_loss += float(kld)
        cum_tail_loss += float(mse_tail)

    avg_Y_loss = cum_Y_loss/nExamples
    res_str = "avg_loss: {:3E}, X_loss: {:3E}, Y_loss: {:3E}, KLD: {:3E}, tail_loss: {:3E}".format(
        cum_loss/nExamples, cum_X_loss/nExamples, avg_Y_loss, cum_kld_loss/nExamples, cum_tail_loss/nExamples)
    if validation:
        res_str = "VALIDATION: "+ res_str

    res_str = "\n"+ res_str
    print(res_str)
    return avg_Y_loss


def train(model,train_data,valid_data, nepochs=10, lr=1e-3, kl_lambda=1, kl_tail=1e2, half=False, cuda=True, batch_size=16, num_workers=8):
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    kl_schedule = T.from_numpy(sigmoid_schedule(nepochs))
    if half:
        optimizer = apex.fp16_utils.FP16_Optimizer(T.optim.Adam(model.parameters(),lr=lr))
    else:
        optimizer = T.optim.Adam(model.parameters(),lr=lr)

    if cuda:
        kl_schedule = kl_schedule.cuda()
    for e in range(nepochs):
        print("\nDenoise epoch {}\n=======================\n".format(e), end="")
        # train model
        if e==0:
            verbose = True
        else:
            verbose = False
        avg_Y_loss = train_helper(model, dataloader, optimizer, len(train_data), X_lambda=1,
            Y_lambda=0, cuda=cuda, validation=False, kl_lambda=kl_schedule[e]*kl_lambda, kl_tail=kl_tail, verbose=verbose)
        # validatation model
        model.eval()
        avg_Y_valid_loss = train_helper(model, valid_dataloader, optimizer, len(valid_data), X_lambda=1,
            Y_lambda=0, cuda=cuda, validation=True, kl_lambda=kl_schedule[e]*kl_lambda, kl_tail=kl_tail, verbose=False)
        model.train()

    sanity_params = list(model.module.parameters())[0]
    model.module.freeze()
    for e in range(nepochs):
        print("\nDenoise epoch {}\n=======================\n".format(e), end="")
        # train model
        avg_Y_loss = train_helper(model, dataloader, optimizer, len(train_data), X_lambda=0,
            Y_lambda=1, cuda=cuda, validation=False, kl_lambda=kl_schedule[e]*kl_lambda, kl_tail=kl_tail, verbose=verbose)
        # validatation model
        model.eval()
        avg_Y_valid_loss = train_helper(model, valid_dataloader, optimizer, len(valid_data), X_lambda=0,
            Y_lambda=1, cuda=cuda, validation=True, kl_lambda=kl_schedule[e]*kl_lambda, kl_tail=kl_tail, verbose=False)
        model.train()
    assert np.all(sanity_params == list(model.module.parameters())[0])

    return avg_Y_loss, avg_Y_valid_loss

def trainBoth(model,train_data,valid_data, nepochs=10, lr=1e-3, kl_lambda=1, kl_tail=1e2, half=False, cuda=True, batch_size=16, num_workers=8):
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    kl_schedule = T.from_numpy(sigmoid_schedule(nepochs))
    if half:
        optimizer = apex.fp16_utils.FP16_Optimizer(T.optim.Adam(model.parameters(),lr=lr))
    else:
        optimizer = T.optim.Adam(model.parameters(),lr=lr)

    if cuda:
        kl_schedule = kl_schedule.cuda()
    for e in range(nepochs):
        print("\nepoch {}\n=======================\n".format(e), end="")
        # train model
        if e==0:
            verbose = True
        else:
            verbose = False
        avg_Y_loss = train_helper(model, dataloader, optimizer, len(train_data), X_lambda=1,
            Y_lambda=1, cuda=cuda, validation=False, kl_lambda=kl_schedule[e]*kl_lambda, kl_tail=kl_tail, verbose=verbose)
        # validatation model
        model.eval()
        avg_Y_valid_loss = train_helper(model, valid_dataloader, optimizer, len(valid_data), X_lambda=1,
            Y_lambda=1, cuda=cuda, validation=True, kl_lambda=kl_schedule[e]*kl_lambda, kl_tail=kl_tail, verbose=False)
        model.train()

    return avg_Y_loss, avg_Y_valid_loss
