import argparse
from utils.datasets import get_data
from tqdm import tqdm
from tensorboardX import SummaryWriter
from importlib import import_module
from torch.optim import Adam
import torch
from utils.visualizations import *


parse=argparse.ArgumentParser(description='VAEs')

parse.add_argument('--cuda',type=bool,default=True)

parse.add_argument('--dataset',type=str,default='mnist')
parse.add_argument('--batch_size',type=int,default=256)

parse.add_argument('--learning_rate',type=float,default=1e-3)
parse.add_argument('--epochs',type=int,default=200)
parse.add_argument('--vis',type=bool,default=True)

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
    writer=SummaryWriter(log_dir='./runs')


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

    epoch_loss/=train_data[0][0].size(0)
    epoch_bar.set_description('Loss=-{:.4f}'.format(epoch_loss))

    test_epoch_loss=0
    with torch.no_grad():
        for test_x,_ in test_data[1]:
            if args.cuda:
                test_x=test_x.cuda()

            batch_loss=model(test_x)

            test_epoch_loss+=batch_loss*test_x.size(0)
        test_epoch_loss/=test_data[0][0].size(0)


    if args.vis:
        writer.add_scalars('ELBO_Loss',{'Train':epoch_loss,'Test':test_epoch_loss},epoch)

        #reconstruct images
        writer.add_image('reconstruct_images',get_reconstruct_images(model,test_data[0]),epoch)





if args.vis:
    writer.close()






