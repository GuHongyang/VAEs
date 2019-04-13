import argparse
from utils.mnist import *
from tqdm import tqdm
from visdom import Visdom
from importlib import import_module

parse=argparse.ArgumentParser(description='VAEs')

parse.add_argument('--batch_size',type=int,default=256)
parse.add_argument('--dataset',type=str,default='mnist')
parse.add_argument('--epochs',type=int,default=200)
parse.add_argument('--vis',type=bool,default=False)
parse.add_argument('--cuda',type=bool,default=True)
parse.add_argument('--model',type=str,default='VAE')

args=parse.parse_args()

data_module=import_module('utils.'+args.dataset)
train_data,test_data=data_module.get_data(args)

model_module=import_module('models.'+args.model)
model=model_module.make_model(args)





epoch_bar=tqdm(range(args.epochs))

