import argparse
from utils.datasets import get_data
from utils.visual import *
from tqdm import tqdm
from visdom import Visdom
from importlib import import_module
from torch.optim import Adam

parse=argparse.ArgumentParser(description='VAEs')

parse.add_argument('--cuda',type=bool,default=True)

parse.add_argument('--dataset',type=str,default='mnist')
parse.add_argument('--batch_size',type=int,default=256)

parse.add_argument('--learning_rate',type=float,default=1e-3)
parse.add_argument('--epochs',type=int,default=200)
parse.add_argument('--vis',type=bool,default=False)

parse.add_argument('--model',type=str,default='VAE')
parse.add_argument('--input_dim',type=int,default=784)
parse.add_argument('--latent_dim',type=int,default=50)
parse.add_argument('--hid_dims',type=list,default=[400,])

args=parse.parse_args()

train_data,test_data=get_data(args)

model_module=import_module('models.'+args.model)
model=model_module.make_model(args)

opti=Adam(model.parameters(),lr=args.learning_rate)

if args.vis:
    vis=Visdom()
    elbo_lines=Line(opts={'xlabel':'Epoch','ylabel':'ELBO'})


epoch_bar=tqdm(range(args.epochs))

for epoch in epoch_bar:
    epoch_loss=0

    batch_bar=tqdm(train_data[1])
    for train_x,_ in batch_bar:
        if args.cuda:
            train_x=train_x.cuda()

        batch_loss=model(train_x)

        opti.zero_grad()
        batch_loss.backward()
        opti.step()

        batch_bar.set_description('loss=-{:.4f}'.format(batch_loss))
        epoch_loss+=batch_loss*train_x.size(0)

    epoch_loss/=train_data[0].__len__()
    epoch_bar.set_description('Loss=-{:.4f}'.format(epoch_loss))





