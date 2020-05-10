import torch.nn.functional as F
import math
import torch
import torch.nn as nn
class LCONV(nn.Module):
    def __init__(self,kernel_size=1,dilation=1,padding=1,stride=1,filters=None,channel=None,center=True,scale=True,bias=True):
        super(LCONV, self).__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.filters = filters
        self.channel = channel
        #self.inputs = input
        #elf.out_h_w = _pair(self.conv_output_shape(self.inputs[1],self.inputs[2],kernel_size,stride,padding,dilation))
        self.scale_bool=scals
        self.center_bool=center
        self.weight = nn.Parameter(torch.Tensor(self.filters, self.channel,
                                                self.kernel_size,self.kernel_size),requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(filters),requires_grad=True)
        self.scale= nn.Parameter(torch.Tensor(filters),requires_grad=True)
        with torch.no_grad():
            nn.init.kaiming_normal(self.weight, mode='fan_out')
        
        
        
    
    def forward(self, x):
        a,b=x.size(2),x.size(3)
        x = F.unfold(x, (self.kernel_size,self.kernel_size),dilation=self.dilation, padding=self.padding,stride=self.stride)
        x = x.transpose(1,2)
        unfold_mean  =x.mean(-1,keepdim=True)
        unfold_var = x.var(-1,unbiased=False, keepdim=True)
       
        unfold_var = torch.sqrt(unfold_var + 1e-14)
        x = (x - unfold_mean) / unfold_var.expand_as(x)
        x =x @ self.weight.view(self.weight.size(0),-1).t()
        
        x= x + self.bias
        x = x.transpose(1,2) 
        
        h,w  =  self.conv_output_shape(h1=a,w1=b,kernel_size=self.kernel_size,stride=self.stride,pad=self.padding,dilation=self.dilation)
        x = x.view(-1, self.filters, h,w )
        
        return x
        
    def conv_output_shape(self,h1=1,w1=1, kernel_size=1,stride=1,pad=0,dilation=1):
        from math import floor
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor((h1 + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride + 1)
        w = floor((w1 + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride + 1)
        return h, w
    
