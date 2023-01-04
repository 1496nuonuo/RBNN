import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable
from scipy.stats import ortho_group
from utils.options import args





class BinarizeConv2d(nn.Conv2d):

    BinarizeConv2d_count = 0

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.k = torch.tensor([10.]).float()
        self.t = torch.tensor([0.1]).float()
        BinarizeConv2d.BinarizeConv2d_count +=1
        self.ind = BinarizeConv2d.BinarizeConv2d_count
        
        #TODO
        self.cosine_similarity = torch.Tensor([1.]).float().cuda()
        # self.cosine_similarity.data.fill_(1)
        

        self.epoch = -1

        w = self.weight
        self.a, self.b = get_ab(np.prod(w.shape[1:])) #a<b  in*w*h
        R1 = torch.tensor(ortho_group.rvs(dim=self.a)).float().cuda()#生成了a*a的正定矩阵,维数指向量几维
        R2 = torch.tensor(ortho_group.rvs(dim=self.b)).float().cuda()#生成了b*b的正定矩阵
        self.register_buffer('R1', R1) #R1 R2不参与学习
        self.register_buffer('R2', R2)
        self.Rweight = torch.ones_like(w) #返回与w维度相同全1的张量,Rweight表示旋转后

        sw = w.abs().view(w.size(0), -1).mean(-1).float().view(w.size(0), 1, 1).detach()#每个卷积核绝对值以后取平均,N*1*1
        self.alpha = nn.Parameter(sw.cuda(), requires_grad=True)#在gpu上，二值化网络的缩放因子
        self.rotate = nn.Parameter(torch.ones(w.size(0), 1, 1, 1).cuda()*np.pi/2, requires_grad=True)#每个卷积核都是pi/2#在gpu上
        self.Rotate = torch.zeros(1)

    def forward(self, input):
        a0 = input
        w = self.weight
        w1 = w - w.mean([1,2,3], keepdim=True)
        w2 = w1 / w1.std([1,2,3], keepdim=True)
        a1 = a0 - a0.mean([1,2,3], keepdim=True)
        a2 = a1 / a1.std([1,2,3], keepdim=True) #标准化
        a, b = self.a, self.b
        X = w2.view(w.shape[0], a, b) #把w2摊开成N*a*b
        if self.epoch > -1 and self.epoch % args.rotation_update == 0:
            for _ in range(3):
                #* update B
                V = self.R1.t() @ X.detach() @ self.R2 #@运算符，矩阵相乘(不是dot product)
                B = torch.sign(V)
                #* update R1
                D1 = sum([Bi@(self.R2.t())@(Xi.t()) for (Bi,Xi) in zip(B,X.detach())])#每次取一个卷积核，每个卷积核对应项求和
                U1, S1, V1 = torch.svd(D1) #对D1奇异值分解(由于是方阵a*a,故也是特征值分解)
                self.R1 = (V1@(U1.t())) #
                #* update R2
                D2 = sum([(Xi.t())@self.R1@Bi for (Xi,Bi) in zip(X.detach(),B)])
                U2, S2, V2 = torch.svd(D2)
                self.R2 = (U2@(V2.t()))
        self.Rweight = ((self.R1.t())@X@(self.R2)).view_as(w)
        delta = self.Rweight.detach() - w2
        w3 = w2 + torch.abs(torch.sin(self.rotate)) * delta

        #* binarize
        bw = BinaryQuantize().apply(w3, self.k.to(w.device), self.t.to(w.device))
        self.cosine_similarity = torch.sum(torch.mul(bw,w))/(torch.norm(bw)*torch.norm(w))
        # if self.BinarizeConv2d_count == 1:
        #     self.writer.add_scalar('/output/logs/cosine_similarity' , self.cosine_similarity.item(), self.epoch)
        #     self.writer.add_scalar('/output/logs/rotate' , torch.abs(torch.sin(self.rotate)[0]).detach().item(), self.epoch)

        if args.a32:
            ba = a2
        else:
            ba = BinaryQuantize_a().apply(a2, self.k.to(w.device), self.t.to(w.device))
        #* 1bit conv
        output = F.conv2d(ba, bw, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        #* scaling factor
        output = output * self.alpha
        return output


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()#the chain rule
        return grad_input, None, None


class BinaryQuantize_a(Function):#二值化激活函数的输出 k=1 t<=1 
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        k = torch.tensor(1.).to(input.device)
        t = max(t, torch.tensor(1.).to(input.device))
        grad_input = k * (2 * torch.sqrt(t**2 / 2) - torch.abs(t**2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input, None, None


def get_ab(N):
    sqrt = int(np.sqrt(N))
    for i in range(sqrt, 0, -1):
        if N % i == 0:
            return i, N // i
