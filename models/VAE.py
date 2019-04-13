import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

def make_model(args):
    return VAE(args)


class Encoder(nn.Module):
    def __init__(self, hid_dims=[400, ], input_dim=784, latent_dim=50):
        super(Encoder,self).__init__()

        self.hid_layers=nn.ModuleList()
        pre_dim=input_dim
        for i in range(len(hid_dims)):
            self.hid_layers.append(nn.Sequential(
                nn.Linear(pre_dim, hid_dims[i]),
                nn.ReLU()
            ))
            pre_dim=hid_dims[i]

        self.z_mu_layer=nn.Linear(pre_dim,latent_dim)
        self.z_sigma_layer=nn.Linear(pre_dim,latent_dim)

    def forward(self, x):
        pre_hid=x
        hid=[]
        for i in range(len(self.hid_layers)):
            hid=self.hid_layers[i](pre_hid)
            pre_hid=hid

        z_mu=self.z_mu_layer(hid)
        z_sigma=self.z_sigma_layer(hid).exp()

        return z_mu, z_sigma


class Decoder(nn.Module):
    def __init__(self, hid_dims=[400, ], input_dim=784, latent_dim=50):
        super(Decoder,self).__init__()

        self.hid_layers=nn.ModuleList()
        pre_dim=latent_dim
        for i in range(len(hid_dims)):
            self.hid_layers.append(nn.Sequential(
                nn.Linear(pre_dim, hid_dims[-i - 1]),
                nn.ReLU()
            ))
            pre_dim=hid_dims[-i - 1]

        self.x_layer=nn.Linear(pre_dim,input_dim)

    def forward(self, z):
        pre_hid=z
        hid = []
        for i in range(len(self.hid_layers)):
            hid=self.hid_layers[i](pre_hid)
            pre_hid=hid

        x_pro=F.sigmoid(self.x_layer(hid))

        return x_pro


class VAE(nn.Module):
    def __init__(self,args):
        super(VAE,self).__init__()
        self.encoder=Encoder(args.hid_dims,args.input_dim,args.latent_dim)
        self.decoder=Decoder(args.hid_dims,args.input_dim,args.latent_dim)
        if args.cuda:
            self.cuda()

    def forward(self,x,L=1):
        z_mu,z_sigma=self.encoder(x)

        KL_div=-0.5*(1+2*z_sigma.log()-z_mu.pow(2)-z_sigma.pow(2))

        rec_loss=0.0
        for l in range(L):
            z = reparameterize(z_mu, z_sigma)
            x_rec = self.decoder(z)
            rec_loss+=F.binary_cross_entropy(x_rec,x,reduce=False)
        rec_loss/=L

        return torch.mean(torch.sum(KL_div,1)+torch.sum(rec_loss,1))

    def reconstruct_x(self,x):
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize(z_mu, z_sigma)
        return self.decoder(z)


def reparameterize(mu,sigma):
    return torch.randn_like(mu)*sigma+mu











