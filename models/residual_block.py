from collections import OrderedDict
import torch
import torch.nn as nn


'''
# --------------------------------------------
# Advanced nn.Sequential
# --------------------------------------------
'''


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


'''
# --------------------------------------------
# basic block
# --------------------------------
# conv + batch normalization + ELU
# --------------------------------------------
'''


class basic_block(nn.Module):
    def __init__(self, in_nc=64, out_nc=64):
        super().__init__()
        self.conv = nn.Conv2d(in_nc, out_nc,
                              kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(
            out_nc, momentum=0.9, eps=1e-04, affine=True)
        self.ELU = nn.ELU(alpha=1.0, inplace=True)

    def forward(self, X):
        Y = self.ELU(self.bn(self.conv(X)))
        return Y


'''
# --------------------------------------------
# residual block
# --------------------------------------------
'''


class residual_block(nn.Module):
    # block_number的可能取值有:1|2|4|8|16
    def __init__(self, in_nc=3, out_nc=3, nc=64, block_number=4):
        super().__init__()
        self.block_number = block_number
        self.have_conv = False
        if block_number == 1:
            self.bb = basic_block(in_nc, out_nc)
            if in_nc != out_nc:
                self.have_conv = True
                self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=1)
        elif block_number == 2:
            self.bb1 = basic_block(in_nc, nc)
            self.bb2 = basic_block(nc, out_nc)
            if in_nc != out_nc:
                self.have_conv = True
                self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=1)
        elif block_number == 4:
            self.bb1 = basic_block(in_nc, nc)
            self.bb2 = basic_block(nc, nc)
            self.bb3 = basic_block(nc, nc)
            self.bb4 = basic_block(nc, out_nc)
            if in_nc != out_nc:
                self.have_conv = True
                self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=1)
        elif block_number == 8:
            self.bb1 = basic_block(in_nc, nc)
            self.bb2 = basic_block(nc, nc)
            self.bb3 = basic_block(nc, nc)
            self.bb4 = basic_block(nc, nc)
            self.bb5 = basic_block(nc, nc)
            self.bb6 = basic_block(nc, nc)
            self.bb7 = basic_block(nc, nc)
            self.bb8 = basic_block(nc, out_nc)
            if in_nc != out_nc:
                self.have_conv = True
                self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=1)

    def forward(self, X):
        if self.block_number == 1:
            Y = self.bb(X)
            if self.have_conv:
                X = self.conv(X)
        elif self.block_number == 2:
            Y = self.bb1(X)
            Y = self.bb2(Y)
            if self.have_conv:
                X = self.conv(X)
        elif self.block_number == 4:
            Y = self.bb1(X)
            Y = self.bb2(Y)
            Y = self.bb3(Y)
            Y = self.bb4(Y)
            if self.have_conv:
                X = self.conv(X)
        elif self.block_number == 8:
            Y = self.bb1(X)
            Y = self.bb2(Y)
            Y = self.bb3(Y)
            Y = self.bb4(Y)
            Y = self.bb5(Y)
            Y = self.bb6(Y)
            Y = self.bb7(Y)
            Y = self.bb8(Y)
            if self.have_conv:
                X = self.conv(X)
        return X-Y
