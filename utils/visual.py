import torch
import numpy as np


class Line():
    def __init__(self,opts,vis):
        self.vis=vis
        self.win=vis.line([0],[0],opts=opts)
        self.y=[]

    def __call__(self,ys):
        self.y.append(ys)
        self.win=self.vis.line(ys,X=np.arange(len(ys)),win=self.win,update='replace')


class Images():
    def __init__(self,opts,nrow,vis):
        self.nrow=nrow
        self.vis=vis
        self.win=vis.images(torch.zeros([100,1,10,10]),nrow=10,opts=opts)

    def __call__(self,images):
        self.win=self.vis.images(images,win=self.win,nrow=self.nrow)
