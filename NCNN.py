import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor

class NCONV(nn.Module):
    def __init__(self, kernel_size=1, dilation=1, padding=1, stride=1,
                 filters=None, channel=None, center=True, scale=True, bias=True):
        super(LCONV, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.filters = filters
        self.channel = channel
        self.center_bool = center
        self.scale_bool = scale

        # Define weights and parameters
        self.weight = nn.Parameter(torch.Tensor(filters, channel, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(filters)) if bias else None
        self.scale = nn.Parameter(torch.Tensor(filters)) if scale else None

        # Initialize weights
        nn.init.kaiming_normal_(self.weight, mode='fan_out')

    def forward(self, x):
        h1, w1 = x.size(2), x.size(3)

        # Apply unfold operation
        x = F.unfold(x, (self.kernel_size, self.kernel_size),
                     dilation=self.dilation, padding=self.padding, stride=self.stride)
        x = x.transpose(1, 2)  # (batch, num_patches, patch_size)

        # Compute mean and variance, then normalize
        if self.center_bool:
            mean = x.mean(dim=-1, keepdim=True)
            x = x - mean
        if self.scale_bool:
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            x = x / torch.sqrt(var + 1e-14)

        # Apply weights
        x = x.matmul(self.weight.view(self.weight.size(0), -1).t())

        # Add bias
        if self.bias is not None:
            x = x + self.bias

        # Reshape back to original dimensions
        x = x.transpose(1, 2)
        h, w = self.conv_output_shape(h1, w1, self.kernel_size, self.stride, self.padding, self.dilation)
        x = x.view(-1, self.filters, h, w)

        return x

    @staticmethod
    def conv_output_shape(h1, w1, kernel_size, stride, pad, dilation):
        """ Compute the output shape after convolution operation """
        h = floor((h1 + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1)
        w = floor((w1 + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1)
        return h, w
