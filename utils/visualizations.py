import torch
from torchvision.utils import make_grid
import numpy as np


def get_reconstruct_images(model,dataset):
    rec_images=torch.Tensor(200,1,28,28)

    for i in range(10):
        ind_i=torch.nonzero(dataset[1]==i)
        rec_images[20*i:20*i+10,:]=dataset[0][ind_i[:10],:].view(-1,1,28,28)
        with torch.no_grad():
            rec_=model.reconstruct_x(dataset[0][ind_i[:10],:].view(-1,784).cuda())
        rec_images[20*i+10:20*(i+1),:]=rec_.cpu().detach().view(-1,1,28,28)

    return make_grid(rec_images,nrow=10)


def interpolation_images(model):

    x,y=np.meshgrid(np.linspace(-3,3,20),np.linspace(-3,3,20))
    z_inter=torch.from_numpy(np.concatenate([np.reshape(x,[-1,1]),np.reshape(y,[-1,1])],1)).cuda().float()
    rec_x=model.decoder(z_inter).cpu().detach()

    return make_grid(rec_x.view(-1,1,28,28),nrow=20)



