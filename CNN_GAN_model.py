import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ops import *


        h4 = F.relu(batch_norm_2d(self.h3(h3)))  #([72, 128, 16, 1])
        h4 = conv_cond_concat(h4,yb)  #([72, 141, 16, 1])
        h4 = conv_prev_concat(h4,h0_prev) #([72, 157, 16, 1])

        g_x = torch.sigmoid(self.h4(h4)) #([72, 1, 16, 128])

        return g_x


class discriminator(nn.Module):
    def __init__(self,pitch_range):
        super(discriminator, self).__init__()

        self.df_dim = 64
        self.dfc_dim = 1024
        self.y_dim = 13

        self.h0_prev = nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(2,pitch_range), stride=(2,2))
        #out channels = y_dim +1 
        self.h1_prev = nn.Conv2d(in_channels=27, out_channels=77, kernel_size=(4,1), stride=(2,2))
        # out channels = df_dim + y_dim
        self.linear1 = nn.Linear(244,self.dfc_dim)
        self.linear2 = nn.Linear(1037,1)

    def forward(self,x,y,batch_size,pitch_range):        

        yb = y.view(batch_size,self.y_dim, 1, 1)
        x = conv_cond_concat(x, yb)  #x.shape torch.Size([72, 14, 16, 128])
        
        h0 = lrelu(self.h0_prev(x),0.2)
        fm = h0
        h0 = conv_cond_concat(h0, yb) #torch.Size([72, 27, 8, 1])

        h1 = lrelu(batch_norm_2d(self.h1_prev(h0)),0.2)  #torch.Size([72, 77, 3, 1])
        h1 = h1.view(batch_size, -1)  #torch.Size([72, 231])
        h1 = torch.cat((h1,y),1)  #torch.Size([72, 244])

        h2 = lrelu(batch_norm_1d(self.linear1(h1)))
        h2 = torch.cat((h2,y),1)  #torch.Size([72, 1037])

        h3 = self.linear2(h2)
        h3_sigmoid = torch.sigmoid(h3)


        return h3_sigmoid, h3, fm
