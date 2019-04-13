import argparse

parse=argparse.ArgumentParser(description='VAEs')

parse.add_argument('--batch_size',type=int,default=256)
parse.add_argument('--dataset',type=str,default='mnist')
parse.add_argument('--epochs',type=int,default=200)
parse.add_argument('--vis',type=bool,default=False)
parse.add_argument('--cuda',type=bool,default=True)


