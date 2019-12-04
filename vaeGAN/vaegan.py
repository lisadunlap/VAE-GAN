from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
from data.mnist import Net
from utils import idx2onehot

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(794, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.mu_ = nn.Sequential(
            #28x28->12x12
            nn.Conv2d(1,8,5,2,0,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #12x12->4x4
            nn.Conv2d(8,64,5,2,0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #4x4->1x1: 20,1,1
            nn.Conv2d(64,20,4,1,0,bias=False),
            nn.ReLU(True)
            )


        self.logsigma_ = nn.Sequential(
            #28x28->12x12
            nn.Conv2d(1,8,5,2,0,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #12x12->4x4
            nn.Conv2d(8,64,5,2,0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #4x4->1x1: 20,1,1
            nn.Conv2d(64,20,4,1,0,bias=False),
            nn.ReLU(True)
            )
        

        self.dec_ = nn.Sequential(
            #1x1->4x4
            nn.ConvTranspose2d(20,20*8,4,1,0,bias=False),  #(ic,oc,kernel,stride,padding)
            nn.BatchNorm2d(20*8), 
            nn.ReLU(True),
            nn.ConvTranspose2d(20*8,20*16,4,2,1,bias=False), #4x4->8x8
            nn.BatchNorm2d(20*16),
            nn.ReLU(True),
            nn.ConvTranspose2d(20*16,20*32,4,2,1,bias=False), #8x8->16x16
            nn.BatchNorm2d(20*32),
            nn.ReLU(True),
            nn.ConvTranspose2d(20*32,1,2,2,2,bias=False), #16x16->28x28
            nn.Sigmoid()
            )
        

    def encode(self, x, y, label, alpha):
        y = idx2onehot(y, 10, label, alpha)
        con = torch.cat((x, y), dim=-1)
        h1 = self.relu(self.fc1(con))
        return self.fc21(h1), self.fc22(h1)

    #def encode_new(self,x):
    #    return self.mu_(x), self.logsigma_(x)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu


    #def decode_new(self,z, y):
    #    z = z.view(-1,z.size(1),1,1)
    #    return(self.dec_(z, y))


    #def decode(self, z, y):
    #    z = z.view(-1,20)
    #    y_c = self.to_categrical(y)
    #    cat = torch.cat((z, y_c), dim=-1)
    #    h3 = self.relu(self.fc3(cat))
    #    ret = self.sigmoid(self.fc4(h3))
    #    ret = torch.narrow(ret, 1, 0, 784)
    #    return ret

    def dec_params(self):
        return self.fc3, self.fc4

    def return_weights(self):
        return self.fc3.weight, self.fc4.weight

    def forward(self, x, y, label=None, alpha = 1):
       #mu, logvar = self.encode_new(x.view(-1, 1,28,28))
       mu, logvar = self.encode(x.view(-1,28*28), y, label, alpha)
       z = self.reparameterize(mu, logvar)
       mu.unsqueeze_(-1)
       logvar.unsqueeze_(-1)
       mu.unsqueeze_(-1)
       logvar.unsqueeze_(-1)
       return mu,logvar
        #return mu,logvar

class Aux(nn.Module):
    def __init__(self):
        super(Aux,self).__init__()

        self.fc3 = nn.Linear(30,400)
        self.fc4 = nn.Linear(400,784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def decode(self,z, y, label, alpha):
        z = z.view(-1,20)
        y_c = idx2onehot(y, 10, label, alpha)
        cat = torch.cat((z, y_c), dim=-1)
        h3 = self.relu(self.fc3(cat))
        ret = self.sigmoid(self.fc4(h3))
        ret = torch.narrow(ret, 1, 0, 784)
        return ret
    
    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps
        else:
          return mu

    def dec_params(self):
        return self.fc3,self.fc4

    def return_weights(self):
        return self.fc3.weight, self.fc4.weight

    
    def forward(self,z, y, label=None, alpha = 1):
        #self.fc3.weight = fc3_weight
        #self.fc4.weight = fc4_weight
        
        #z = self.reparameterize(mu,logvar)
        #other.fc3,other.fc4 = self.dec_params()
        #return self.decode(z).view(-1,28,28)
        return self.decode(z, y, label, alpha)


    

class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        
        self.main = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self,x, y):
        x = x.view(x.size(0), 784)
        c = self.label_emb(y)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        dl = self.main(out)
        return out, dl

def loss_function(recon_x, x, mu, logvar,bsz=100):
    #BCE = F.binary_cross_entropy(recon_x.view(-1,784), x.view(-1, 784))
    #MSE = F.mse_loss(recon_x.view(-1,784), x.view(-1,784))
    MSE = F.mse_loss(recon_x,x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= bsz * 784

    return MSE + KLD