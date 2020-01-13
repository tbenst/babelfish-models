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
import mlflow



class DeepSkip(Vol2D):
    def __init__(self, nZ=11, H=232, W=512, nEmbedding=20, prev_frames=1, next_frames=1,
                 pred_hidden=20, tensor=T.cuda.FloatTensor):
        super(DeepSkip, self).__init__(tensor)
        self.tensor = tensor
        self.nZ = nZ
        self.H = H
        self.W = W
        self.lowH = 16
        self.lowW = 16
        self.lowFeatures = 1
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
        self.nhalf_embed = int(self.nEmbedding/2)
        # Prediction
        self.pred1 = nn.Linear(self.nhalf_embed, pred_hidden) # add dim for shock_{t+1}
        self.pred2 = nn.Linear(pred_hidden, self.nhalf_embed) # last 10 (context) are unused

        # Prediction
        self.predz1 = nn.Linear(self.nhalf_embed, pred_hidden) # add dim for shock_{t+1}
        self.predz2 = nn.Linear(pred_hidden, self.nhalf_embed) # last 10 (context) are unused

        # Decoding
        self.activation = nn.Tanh()
        # only use 10 embeddings for frame decoding, the other 10 are context
        self.decoding = nn.Linear(self.nhalf_embed, self.lowFeatures*nZ*self.lowH*self.lowW)
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
        self._initialize_weights()

    def _initialize_weights(self):
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

    def encode(self, x):
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
        return {"mean": mean, "logvar": logvar, "layer_outputs": layer_outputs}

    def predict(self, x):
        x = self.activation(self.pred1(x))
        x = self.pred2(x)
        return x

    def predictZero(self, x):
        x = self.activation(self.predz1(x))
        x = self.predz2(x)
        return x

    def decode(self, x, layer_output):
        # b x 10
        # only use first half for brain data
        x = self.activation(self.decoding(x[:,:int(self.nEmbedding/2)]))
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
        return x

    def forward(self, x):
        """Return Previous volume (denoised), next volume (prediction), latent mean and logvar.
        
        x is B x T x Z x H x W
        """

        output = self.encode(x)
        mean = output["mean"]
        logvar = output["logvar"]
        layer_outputs = output["layer_outputs"]
        encoded = self.sample_embedding(mean, logvar)
        encoded_prev = self.predictZero(encoded[:,self.nhalf_embed:])
        encoded_pred = self.predict(encoded[:,:self.nhalf_embed])
        prev = self.decode(encoded_prev, layer_outputs) # force to use only skip connections for decode
        pred = self.decode(encoded_pred, layer_outputs)
        return {"prev": prev, "pred": pred, "mean": mean,
            "logvar": logvar} # should we move variational layer? or return encoded_pred?

def unit_norm_KL_divergence(mu, logvar):
    "Reconstruction + KL divergence losses summed over all elements and batch."
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp())


def train(model,train_data,valid_data, nepochs=10, lr=1e-3, kl_lambda=1, half=False, cuda=True, batch_size=16, num_workers=8):
    # TODO allow specifying device
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    kl_schedule = T.from_numpy(sigmoid_schedule(nepochs))
    if half:
        optimizer = apex.fp16_utils.FP16_Optimizer(T.optim.Adam(model.parameters(),lr=lr))
    else:
        optimizer = T.optim.Adam(model.parameters(),lr=lr)

    if cuda:
        kl_schedule = kl_schedule.cuda()
    for e in range(nepochs):
        print("epoch {}: ".format(e), end="")
        cum_loss = 0
        cum_X_loss = 0
        cum_Y_loss = 0
        cum_kld_loss = 0
        i = 0
        for batch_data in tqdm(dataloader):
            X, Y = batch_data
            # TODO refactor for generic auxiliary vars
            X = X["brain"]
            Y = Y["brain"]
            if cuda:
                # TODO data.to('cuda:0', non_blocking=True) or similar
                X = X.cuda()
                Y = Y.cuda()
            output = model(X)
            X_pred = output["prev"]
            Y_pred = output["pred"]
            mean = output["mean"]
            logvar = output["logvar"]
            if half:
                X_pred = X_pred.float()
                Y_pred = Y_pred.float()
                mean = mean.float()
                logvar = logvar.float()
            kld = unit_norm_KL_divergence(mean, logvar)
            mse_X = F.mse_loss(X_pred, Y[:,0])
            mse_Y = F.mse_loss(Y_pred, Y[:,-1])
            loss = mse_X + mse_Y + kl_lambda*kl_schedule[e] * kld
            if e==0:
                mlflow.log_metrics({
                    "MSE_X": float(mse_X)/batch_size,
                    "MSE_Y": float(mse_Y)/batch_size,
                    "KLD": float(kld)/batch_size,
                    }, step=0)
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
            i+=1
        
        avg_loss = cum_loss/len(train_data)
        avg_Y_loss = cum_Y_loss/len(train_data)
        avg_X_loss = cum_X_loss/len(train_data)
        avg_KLD_loss = cum_kld_loss/len(train_data)
        mlflow.log_metrics({
            "avg_loss": avg_loss,
            "MSE_X": avg_X_loss,
            "MSE_Y": avg_Y_loss,
            "KLD": avg_KLD_loss
            }, step=e)
        print("avg_loss: {:3E}, X_loss: {:3E}, Y_loss: {:3E}, KLD: {:3E}".format(
            avg_loss, avg_X_loss, avg_Y_loss, avg_KLD_loss))
        mlflow.pytorch.log_model(model, f"models/epoch/{e}")
    return avg_X_loss, avg_Y_loss


def validation_loss(model,valid_data, kl_lambda=1, half=False, cuda=True, batch_size=16, num_workers=8):
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True)
    cum_loss = 0
    cum_X_loss = 0
    cum_Y_loss = 0
    cum_kld_loss = 0
    with T.no_grad():
        model.eval()
        for batch_data in tqdm(valid_dataloader):
            X, Y = batch_data
            X = X["brain"]
            Y = Y["brain"]
            if cuda:
                X = X.cuda()
                Y = Y.cuda()
            output = model(X)
            X_pred = output["prev"]
            Y_pred = output["pred"]
            mean = output["mean"]
            logvar = output["logvar"]
            if half:
                X_pred = X_pred.float()
                Y_pred = Y_pred.float()
                mean = mean.float()
                logvar = logvar.float()
            kld = unit_norm_KL_divergence(mean, logvar)
            mse_X = F.mse_loss(X_pred, Y[:,0])
            mse_Y = F.mse_loss(Y_pred, Y[:,-1])
            loss = mse_X + mse_Y + kl_lambda * kld
            cum_loss += float(loss)
            cum_X_loss += float(mse_X)
            cum_Y_loss += float(mse_Y)
            cum_kld_loss += float(kld)
    model.train()
    avg_Y_valid_loss = cum_Y_loss/len(valid_data)
    avg_X_valid_loss = cum_X_loss/len(valid_data)
    print("VALIDATION: avg_loss: {:3E}, X_loss: {:3E}, Y_loss: {:3E}, KLD: {:3E}".format(
    cum_loss/len(valid_data), cum_X_loss/len(valid_data), avg_Y_valid_loss, cum_kld_loss/len(valid_data)))
    return avg_X_valid_loss, avg_Y_valid_loss

def deep_skip_predict(model, batch, mask, plane):
    with T.no_grad():
        model.eval()
        X, Y = batch
        X = X["brain"]
        X = X[None]
        Y = Y["brain"]
        Y = Y[None]
        if cuda:
            X = X.cuda()
            Y = Y.cuda()
        output = model(X)
        X_pred = output["prev"]
        Y_pred = output["pred"]
        mean = output["mean"]
        logvar = output["logvar"]

        if half:
            X_pred = X_pred.float()
            Y_pred = Y_pred.float()
            mean = mean.float()
            logvar = logvar.float()
        tmask = T.from_numpy(mask).cuda()
#         mse_Y = F.mse_loss(Y_pred[0,p]*tmask, Y[0,-1,p]*tmask, size_average=False)
        mse_Y = T.sqrt(T.sum((Y_pred[0,plane]*tmask - Y[0,-1,plane]*tmask)**2))
        model.train()
        return mse_Y, Y_pred
