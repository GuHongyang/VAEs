from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset,DataLoader
from torchvision import transforms


def get_data(args):

    if args.dataset == 'mnist':
        file_path='./data/'+args.dataset
        train_dataset=MNIST(root=file_path,train=True,transform=transforms.ToTensor(),download=True)
        test_dataset=MNIST(root=file_path,train=False,transform=transforms.ToTensor(),download=True)

        train_dataset=TensorDataset(train_dataset.train_data.float().view(-1,784)/255.,train_dataset.train_labels)
        test_dataset=TensorDataset(test_dataset.test_data.float().view(-1,784)/255.,test_dataset.test_labels)

        train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=8)
        test_dataloader=DataLoader(test_dataset,batch_size=args.batch_size)

        return (train_dataset,train_dataloader),(test_dataset,test_dataloader)

    else:
        raise TypeError


