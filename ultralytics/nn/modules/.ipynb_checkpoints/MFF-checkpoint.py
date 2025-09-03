import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ultralytics.nn.modules.conv import *
from ultralytics.nn.modules.attention import *
from ..modules.conv import *

class MFF(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        inter_channels = inc[1]*4
    
        
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)
            
        self.A = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            DWConv(2 * inc[1], 2 * inter_channels),
            nn.BatchNorm2d(2 * inter_channels),
            DWConv(2 * inter_channels, 2 * inc[1]),
            nn.BatchNorm2d(2 * inc[1]),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        x_concat = torch.cat([x0, x1], dim=1)
        x_concat = self.A(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size(1), x1.size(1)], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)
    

