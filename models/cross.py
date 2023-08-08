from ast import arg
from turtle import forward
import torch
import torch.nn.functional as F
import math
from torch.nn import init
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def masked_softmax(src, mask=None, dim=1):
    #out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(src, dim=dim)
    #out = out.masked_fill(~mask, 0)
    return out


class Refine(torch.nn.Module):
    def __init__(self,channel,drop):
        super().__init__()   

        self.nn = nn.Sequential(
            nn.Linear(channel, 128),
            nn.LeakyReLU(),
            nn.Dropout(drop),

            nn.Linear(128, 30),
            nn.LeakyReLU(),
            nn.Dropout(drop),

            nn.Linear(30, 2),
        )
    def forward(self,x, tem = 1):
        B = x.size(0)
        p1 = self.nn(x.mean(1).view((B,-1)))
        return F.log_softmax(p1/tem, 1), F.softmax(p1/tem, 1)

class Cross(torch.nn.Module):
    def __init__(self, in_channel, kernel_size, num_classes=2,args=None):
        super(Cross, self).__init__()
        self.in_planes = 1 
        self.d = kernel_size
        self.batch_norm = nn.BatchNorm2d(2)
        self.channel = args.channel
        self.ll = args.layer
        self.ab = args.ab
        self.drop = 0.2

        num_layers = args.gru

        self.psi_1  = nn.GRU(input_size=self.d, hidden_size=args.channel, num_layers=num_layers)
        self.psi_2  = nn.GRU(input_size=self.d, hidden_size=args.channel, num_layers=num_layers)

        self.mlp1 = nn.Sequential(
            nn.Linear(self.channel * 2, self.channel * 4),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.channel * 4, self.channel * 2),
            nn.ReLU(),
            nn.Dropout(self.drop),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(self.channel * 2, self.channel * 4),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.channel * 4, self.channel * 2),
            nn.ReLU(),
            nn.Dropout(self.drop),
        )
        self.nf_1 = torch.nn.Conv2d(self.channel * 4,self.channel * 4,(self.d,1))

        self.refine1 = Refine(self.channel,self.drop)
        self.refine2 = Refine(self.channel,self.drop)

        self.bn1 = nn.BatchNorm2d(self.channel * 2)
        self.mlp_bn1 = nn.BatchNorm2d(self.channel)

        self.mlp_bn2 = nn.BatchNorm2d(self.channel)
        self.bn2 = nn.BatchNorm2d(self.channel * 2)

        self.dense1 = nn.Linear(self.channel * 2 * 2,128)
        self.dense2 = nn.Linear(128,30)
        self.dense3 = nn.Linear(30,num_classes)
    def forward(self, x, tem =1, get_corr = False):
        #x = self.batch_norm(x)
        B = x.size(0)

        h_s = self.psi_1(x[:,0])[0].squeeze(2) # fmri
        h_t = self.psi_1(x[:,1])[0].squeeze(2) # dti

        h_s = self.mlp_bn1(h_s.permute(0,2,1)[...,None])[...,0].permute(0,2,1)
        h_t = self.mlp_bn2(h_t.permute(0,2,1)[...,None])[...,0].permute(0,2,1)

        S_hat1 = h_s @ h_t.transpose(-1, -2)
        S_hat2 = h_t @ h_s.transpose(-1, -2)
        S_hat0 = (S_hat1 + S_hat2) / 2

        S_00 = masked_softmax(S_hat0)

        r_s = S_00 @ h_t
        r_t = S_00 @ h_s

        h_st1 = torch.cat((h_t, r_s), dim=2)
        h_st2 = torch.cat((h_s, r_t), dim=2)

        x1 = self.mlp1(h_st1)
        x2 = self.mlp2(h_st2)

        x1 = self.bn1(x1.permute(0,2,1)[...,None])[...,0].permute(0,2,1)
        x2 = self.bn2(x2.permute(0,2,1)[...,None])[...,0].permute(0,2,1)
        for i in range(self.ll):
            
            x1 = torch.einsum("npq,nqc->npc", S_00, x1)
            x1 = F.leaky_relu(x1)
            x1 = F.dropout(x1,self.drop,training = self.training)

            x2 = torch.einsum("npq,nqc->npc", S_00, x2)
            x2 = F.leaky_relu(x2)
            x2 = F.dropout(x2,self.drop,training = self.training)

        x = torch.cat((x1, x2), dim=2)
        x = x.permute(0,2,1)
        out = self.nf_1(x[...,None]).view((B,-1))

        out = F.dropout(F.leaky_relu(self.dense1(out)),p=self.drop,training = self.training)
        out = F.dropout(F.leaky_relu(self.dense2(out)),p=self.drop,training = self.training)
        out = F.leaky_relu(self.dense3(out))

        out_p = F.softmax(out,1)

        out1_log, out1_p = self.refine1(h_s)
        out2_log, out2_p = self.refine2(h_t)
        if get_corr is True:
            return out_p, out1_log, out2_log, S_00

        return out_p, out1_log, out2_log