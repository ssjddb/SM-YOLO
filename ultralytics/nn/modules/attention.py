import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.nn import Softmax, init
from .conv import *
from einops import rearrange


##################### FLA #######################
class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_patches=64, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1,
                 focusing_factor=3.0, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        # self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))


    def forward(self, x):
        B, C, H, W = x.shape  # 输入为四维：[批次大小, 通道数, 高度, 宽度]
        dtype, device = x.dtype, x.device
        # 调整输入以匹配原始模块的预期格式
        x = rearrange(x, 'b c h w -> b (h w) c')
        q = self.q(x)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        N = H * W  # 序列长度
        # 重新生成位置编码
        positional_encoding = nn.Parameter(torch.zeros(size=(1, N, self.dim), device=device))
        k = k + positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        bool = False
        if dtype == torch.float16:
            q = q.float()
            k = k.float()
            v = v.float()
            bool = True
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]
        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.permute(0, 2, 1), size=x.shape[1], mode='linear').permute(0, 2, 1)
        if bool:
            v = v.to(torch.float16)
            x = x.to(torch.float16)

        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


######################## MPCA #######################
class MPCA(nn.Module):
    # MultiPath Coordinate Attention
    def __init__(self, channels) -> None:
        super().__init__()
        
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(channels, channels)
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_hw = Conv(channels, channels, (3, 1))
        self.conv_pool_hw = Conv(channels, channels, 1)
    
    def forward(self, x):
        _, _, h, w = x.size()
        x_pool_h, x_pool_w, x_pool_ch = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2), self.gap(x)
        x_pool_hw = torch.cat([x_pool_h, x_pool_w], dim=2)
        x_pool_hw = self.conv_hw(x_pool_hw)
        x_pool_h, x_pool_w = torch.split(x_pool_hw, [h, w], dim=2)
        x_pool_hw_weight = self.conv_pool_hw(x_pool_hw).sigmoid()
        x_pool_h_weight, x_pool_w_weight = torch.split(x_pool_hw_weight, [h, w], dim=2)
        x_pool_h, x_pool_w = x_pool_h * x_pool_h_weight, x_pool_w * x_pool_w_weight
        x_pool_ch = x_pool_ch * torch.mean(x_pool_hw_weight, dim=2, keepdim=True)
        return x * x_pool_h.sigmoid() * x_pool_w.permute(0, 1, 3, 2).sigmoid() * x_pool_ch.sigmoid()


########################## CAFM ###################

class CAFM(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=(1,1,1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=(3,3,3), stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1,1,1), bias=bias)
        self.fc = nn.Conv3d(3*self.num_heads, 9, kernel_size=(1,1,1), bias=True)

        self.dep_conv = nn.Conv3d(9*dim//self.num_heads, dim, kernel_size=(3,3,3), bias=True, groups=dim//self.num_heads, padding=1)


    def forward(self, x):
        b,c,h,w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0,2,3,1) 
        f_all = qkv.reshape(f_conv.shape[0], h*w, 3*self.num_heads, -1).permute(0, 2, 1, 3) 
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        #local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9*x.shape[1]//self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv) # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv

        return output


############### iDRB ###############
class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def switch_to_deploy(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class iDRB(nn.Module):
	def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
				 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=16, window_size=7,
				 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
		super().__init__()
		self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
		self.act = Conv.default_act if act else nn.Identity()
		dim_mid = int(dim_in * exp_ratio)
		self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
		self.attn_s = attn_s
		if self.attn_s:
			assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
			self.dim_head = dim_head
			self.window_size = window_size
			self.num_head = dim_in // dim_head
			self.scale = self.dim_head ** -0.5
			self.attn_pre = attn_pre
			self.qk = nn.Conv2d(dim_in, int(dim_in * 2), 1, bias=qkv_bias)
			self.v = nn.Sequential(
				nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
				self.act
			)
			self.attn_drop = nn.Dropout(attn_drop)
		else:
			if v_proj:
				self.v = nn.Sequential(
					nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
					self.act
				)
			else:
				self.v = nn.Identity()
		self.conv_local = DilatedReparamBlock(dim_mid, dw_ks)
		self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()
		
		self.proj_drop = nn.Dropout(drop)
		self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
		self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
	
	def forward(self, x):
		shortcut = x
		x = self.norm(x)
		B, C, H, W = x.shape
		if self.attn_s:
			# padding
			if self.window_size <= 0:
				window_size_W, window_size_H = W, H
			else:
				window_size_W, window_size_H = self.window_size, self.window_size
			pad_l, pad_t = 0, 0
			pad_r = (window_size_W - W % window_size_W) % window_size_W
			pad_b = (window_size_H - H % window_size_H) % window_size_H
			x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
			n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
			x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
			# attention
			b, c, h, w = x.shape
			qk = self.qk(x)
			qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
			q, k = qk[0], qk[1]
			attn_spa = (q @ k.transpose(-2, -1)) * self.scale
			attn_spa = attn_spa.softmax(dim=-1)
			attn_spa = self.attn_drop(attn_spa)
			if self.attn_pre:
				x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
				x_spa = attn_spa @ x
				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
				x_spa = self.v(x_spa)
			else:
				v = self.v(x)
				v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
				x_spa = attn_spa @ v
				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
			# unpadding
			x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
			if pad_r > 0 or pad_b > 0:
				x = x[:, :, :H, :W].contiguous()
		else:
			x = self.v(x)

		x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
		
		x = self.proj_drop(x)
		x = self.proj(x)
		
		x = (shortcut + self.drop_path(x)) if self.has_skip else x
		return x
################### EMA ##############

import torch
from torch import nn


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class MSCAAttention(nn.Module):
    # SegNext NeurIPS 2022
    # https://github.com/Visual-Attention-Network/SegNeXt/tree/main
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class ChannelAttention(nn.Module):
    # Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class MSAM(nn.Module):
    # Convolutional Block Attention Module
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = MSCAAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


####################### ELA ############################

class ELA(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=3),
            nn.GroupNorm(16, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.conv1x1(self.pool_h(x).reshape((b, c, h))).reshape((b, c, h, 1))
        x_w = self.conv1x1(self.pool_w(x).reshape((b, c, w))).reshape((b, c, 1, w))
        return x * x_h * x_w


########################## LSK ####################################

class LSK(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


########################################################################
# ---------------------------SE Begin---------------------------
class SE(nn.Module):
    def __init__(self, c1, ratio=16):
        super(SE, self).__init__()
        # c*1*1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


# ---------------------------SE End---------------------------


# ---------------------------ECA Begin---------------------------
class ECA(nn.Module):

    def __init__(self, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# ---------------------------ECA End---------------------------


# ---------------------------CA Begin---------------------------
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


# ---------------------------CA End---------------------------


# ---------------------------CBAM Begin---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, c1, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


# ---------------------------CBAM End---------------------------
# ---------------------------SimAM Begin---------------------------
class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
                x_minus_mu_square
                / (
                        4
                        * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
                )
                + 0.5
        )

        return x * self.activaton(y)


# ---------------------------SimAM End---------------------------


# ---------------------------S2-MLPv2 Begin---------------------------
def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, : c // 4] = x[:, : w - 1, :, : c // 4]
    x[:, : w - 1, :, c // 4: c // 2] = x[:, 1:, :, c // 4: c // 2]
    x[:, :, 1:, c // 2: c * 3 // 4] = x[:, :, : h - 1, c // 2: c * 3 // 4]
    x[:, :, : h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, : c // 4] = x[:, :, : h - 1, : c // 4]
    x[:, :, : h - 1, c // 4: c // 2] = x[:, :, 1:, c // 4: c // 2]
    x[:, 1:, :, c // 2: c * 3 // 4] = x[:, : w - 1, :, c // 2: c * 3 // 4]
    x[:, : w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)
        a = torch.sum(torch.sum(x_all, 1), 1)
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))
        hat_a = hat_a.reshape(b, self.k, c)
        bar_a = self.softmax(hat_a)
        attention = bar_a.unsqueeze(-2)
        out = attention * x_all
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c: c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x


# ---------------------------S2-MLPv2 End---------------------------


# ---------------------------NAMAttention Begin---------------------------
class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class NAMAttention(nn.Module):
    def __init__(self, channels):
        super(NAMAttention, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1


# ---------------------------NAMAttention End---------------------------


# ---------------------------Criss-CrossAttention Begin---------------------------
import torch
import torch.nn as nn
from torch.nn import Softmax

def INF(B, H, W, device):
    return (
        -torch.diag(torch.tensor(float("inf"), device=device).repeat(H), 0)
        .unsqueeze(0)
        .repeat(B * W, 1, 1)
    )

class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        device = x.device
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = (
            proj_query.permute(0, 3, 1, 2)
            .contiguous()
            .view(m_batchsize * width, -1, height)
            .permute(0, 2, 1)
        )
        proj_query_W = (
            proj_query.permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * height, -1, width)
            .permute(0, 2, 1)
        )
        proj_key = self.key_conv(x)
        proj_key_H = (
            proj_key.permute(0, 3, 1, 2)
            .contiguous()
            .view(m_batchsize * width, -1, height)
        )
        proj_key_W = (
            proj_key.permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * height, -1, width)
        )
        proj_value = self.value_conv(x)
        proj_value_H = (
            proj_value.permute(0, 3, 1, 2)
            .contiguous()
            .view(m_batchsize * width, -1, height)
        )
        proj_value_W = (
            proj_value.permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * height, -1, width)
        )
        energy_H = (
            torch.bmm(proj_query_H, proj_key_H) + INF(m_batchsize, height, width, device)
        ).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(
            m_batchsize, height, width, width
        )
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = (
            concate[:, :, :, 0:height]
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * width, height, height)
        )
        att_W = (
            concate[:, :, :, height: height + width]
            .contiguous()
            .view(m_batchsize * height, width, width)
        )
        out_H = (
            torch.bmm(proj_value_H, att_H.permute(0, 2, 1))
            .view(m_batchsize, width, -1, height)
            .permute(0, 2, 3, 1)
        )
        out_W = (
            torch.bmm(proj_value_W, att_W.permute(0, 2, 1))
            .view(m_batchsize, height, -1, width)
            .permute(0, 2, 1, 3)
        )
        return self.gamma * (out_H + out_W) + x

# ---------------------------Criss-CrossAttention End---------------------------


# ---------------------------GAMAttention Begin---------------------------
class GAMAttention(nn.Module):

    def __init__(self, c1, c2, group=True, rate=4):
        super(GAMAttention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1),
        )
        self.spatial_attention = nn.Sequential(
            (
                nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate)
                if group
                else nn.Conv2d(c1, int(c1 / rate), kernel_size=7, padding=3)
            ),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            (
                nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate)
                if group
                else nn.Conv2d(int(c1 / rate), c2, kernel_size=7, padding=3)
            ),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        return out


def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out


# ---------------------------GAMAttention End---------------------------


# ---------------------------Selective Kernel Attention Begin-------------------------
class SKAttention(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    channel,
                                    channel,
                                    kernel_size=k,
                                    padding=k // 2,
                                    groups=group,
                                ),
                            ),
                            ("bn", nn.BatchNorm2d(channel)),
                            ("relu", nn.ReLU()),
                        ]
                    )
                )
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


# ---------------------------Selective Kernel Attention End---------------------------


# ---------------------------ShuffleAttention Begin---------------------------
from torch.nn.parameter import Parameter


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


# ---------------------------ShuffleAttention End---------------------------


# ---------------------------A2-Net  Begin---------------------------
class DoubleAttention(nn.Module):

    def __init__(self, in_channels, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        c__ = int(in_channels * 0.5 * 0.5)  # hidden channels
        self.reconstruct = reconstruct
        self.c_m = c_m = c__
        self.c_n = c_n = c__
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        A = self.convA(x)  # b,c_m,h,w
        B = self.convB(x)  # b,c_n,h,w
        V = self.convV(x)  # b,c_n,h,w
        tmpA = A.view(b, self.c_m, -1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1), dim=1)
        attention_vectors = F.softmax(V.view(b, self.c_n, -1), dim=1)
        # step 1: feature gating
        global_descriptors = torch.bmm(
            tmpA, attention_maps.permute(0, 2, 1)
        )  # b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors)  # b,c_m,h*w
        tmpZ = tmpZ.view(b, self.c_m, h, w)  # b,c_m,h,w
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)

        return tmpZ


# ---------------------------A2-Net  End---------------------------


# ---------------------------RFB  Begin---------------------------
class BasicConv(nn.Module):

    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            relu=True,
            bn=True,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(
            self,
            in_planes,
            out_planes,
            stride=1,
            scale=0.1,
            map_reduce=8,
            vision=1,
            groups=1,
    ):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(
                in_planes,
                inter_planes,
                kernel_size=1,
                stride=1,
                groups=groups,
                relu=False,
            ),
            BasicConv(
                inter_planes,
                2 * inter_planes,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=groups,
            ),
            BasicConv(
                2 * inter_planes,
                2 * inter_planes,
                kernel_size=3,
                stride=1,
                padding=vision + 1,
                dilation=vision + 1,
                relu=False,
                groups=groups,
            ),
        )
        self.branch1 = nn.Sequential(
            BasicConv(
                in_planes,
                inter_planes,
                kernel_size=1,
                stride=1,
                groups=groups,
                relu=False,
            ),
            BasicConv(
                inter_planes,
                2 * inter_planes,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=groups,
            ),
            BasicConv(
                2 * inter_planes,
                2 * inter_planes,
                kernel_size=3,
                stride=1,
                padding=vision + 2,
                dilation=vision + 2,
                relu=False,
                groups=groups,
            ),
        )
        self.branch2 = nn.Sequential(
            BasicConv(
                in_planes,
                inter_planes,
                kernel_size=1,
                stride=1,
                groups=groups,
                relu=False,
            ),
            BasicConv(
                inter_planes,
                (inter_planes // 2) * 3,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
            ),
            BasicConv(
                (inter_planes // 2) * 3,
                2 * inter_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            ),
            BasicConv(
                2 * inter_planes,
                2 * inter_planes,
                kernel_size=3,
                stride=1,
                padding=vision + 4,
                dilation=vision + 4,
                relu=False,
                groups=groups,
            ),
        )

        self.ConvLinear = BasicConv(
            6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False
        )
        self.shortcut = BasicConv(
            in_planes, out_planes, kernel_size=1, stride=stride, relu=False
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


# ---------------------------RFB  End---------------------------


# ---------------------------CoTAttention Begin---------------------------
class CoTAttention(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=4,
                bias=False,
            ),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1),
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w

        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


# ---------------------------CoTAttention End---------------------------


# ---------------------------EffectiveSEModule Begin---------------------------
from timm.layers.create_act import create_act_layer


class EffectiveSEModule(nn.Module):
    def __init__(self, channels, add_maxpool=False, gate_layer="hard_sigmoid"):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


# ---------------------------EffectiveSEModule End---------------------------


# ---------------------------GlobalContext Begin---------------------------
from timm.layers.norm import LayerNorm2d


class GlobalContext(nn.Module):

    def __init__(
            self,
            channels,
            use_attn=True,
            fuse_add=False,
            fuse_scale=True,
            init_last_zero=False,
            rd_ratio=1.0 / 8,
            rd_channels=None,
            rd_divisor=1,
            act_layer=nn.ReLU,
            gate_layer="sigmoid",
    ):
        super(GlobalContext, self).__init__()
        act_layer = get_act_layer(act_layer)

        self.conv_attn = (
            nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
        )

        if rd_channels is None:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        if fuse_add:
            self.mlp_add = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_scale = None

        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(
                self.conv_attn.weight, mode="fan_in", nonlinearity="relu"
            )
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)

        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x


# ---------------------------GlobalContext End---------------------------

# ---------------------------GatherExcite Begin---------------------------
import math
from timm.layers.create_act import create_act_layer, get_act_layer
from timm.layers.create_conv2d import create_conv2d
from timm.layers.helpers import make_divisible
from timm.layers.mlp import ConvMlp


class GatherExcite(nn.Module):
    def __init__(
            self,
            channels,
            feat_size=None,
            extra_params=False,
            extent=0,
            use_mlp=True,
            rd_ratio=1.0 / 16,
            rd_channels=None,
            rd_divisor=1,
            add_maxpool=False,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            gate_layer="sigmoid",
    ):
        super(GatherExcite, self).__init__()
        self.add_maxpool = add_maxpool
        act_layer = get_act_layer(act_layer)
        self.extent = extent
        if extra_params:
            self.gather = nn.Sequential()
            if extent == 0:
                assert (
                        feat_size is not None
                ), "spatial feature size must be specified for global extent w/ params"
                self.gather.add_module(
                    "conv1",
                    create_conv2d(
                        channels,
                        channels,
                        kernel_size=feat_size,
                        stride=1,
                        depthwise=True,
                    ),
                )
                if norm_layer:
                    self.gather.add_module(f"norm1", nn.BatchNorm2d(channels))
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    self.gather.add_module(
                        f"conv{i + 1}",
                        create_conv2d(
                            channels, channels, kernel_size=3, stride=2, depthwise=True
                        ),
                    )
                    if norm_layer:
                        self.gather.add_module(f"norm{i + 1}", nn.BatchNorm2d(channels))
                    if i != num_conv - 1:
                        self.gather.add_module(f"act{i + 1}", act_layer(inplace=True))
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent

        if not rd_channels:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )
        self.mlp = (
            ConvMlp(channels, rd_channels, act_layer=act_layer)
            if use_mlp
            else nn.Identity()
        )
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        size = x.shape[-2:]
        if self.gather is not None:
            x_ge = self.gather(x)
        else:
            if self.extent == 0:
                # global extent
                x_ge = x.mean(dim=(2, 3), keepdims=True)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * x.amax((2, 3), keepdim=True)
            else:
                x_ge = F.avg_pool2d(
                    x,
                    kernel_size=self.gk,
                    stride=self.gs,
                    padding=self.gk // 2,
                    count_include_pad=False,
                )
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(
                        x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2
                    )
        x_ge = self.mlp(x_ge)
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)


# ---------------------------GatherExcite End---------------------------


# ---------------------------MHSA Begin---------------------------
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(
                torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                requires_grad=True,
            )
            self.rel_w_weight = nn.Parameter(
                torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                requires_grad=True,
            )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            content_position = (
                (self.rel_h_weight + self.rel_w_weight)
                .view(1, self.heads, C // self.heads, -1)
                .permute(0, 1, 3, 2)
            )  # 1,4,1024,64

            content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
            content_position = (
                content_position
                if (content_content.shape == content_position.shape)
                else content_position[
                     :,
                     :,
                     :c3,
                     ]
            )
            assert content_content.shape == content_position.shape
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out


# ---------------------------MHSA End---------------------------


# ---------------------------ParNetAttention Begin---------------------------
class ParNetAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid(),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1), nn.BatchNorm2d(channel)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)
        return y


# ---------------------------ParNetAttention End---------------------------


# ---------------------------ParallelPolarizedSelfAttention Begin---------------------------
class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = (
            self.sigmoid(
                self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))
            )
            .permute(0, 2, 1)
            .reshape(b, c, 1, 1)
        )  # bs,c,1,1
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(x)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(x)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * x
        out = spatial_out + channel_out
        return out


# ---------------------------ParallelPolarizedSelfAttention End---------------------------


# ---------------------------SpatialGroupEnhance Begin---------------------------
class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=8):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


# ---------------------------SpatialGroupEnhance End---------------------------


# ---------------------------SequentialPolarizedSelfAttention Begin---------------------------
class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = (
            self.sigmoid(
                self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))
            )
            .permute(0, 2, 1)
            .reshape(b, c, 1, 1)
        )  # bs,c,1,1
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(channel_out)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(channel_out)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * channel_out
        return spatial_out


# ---------------------------SequentialPolarizedSelfAttention End---------------------------


# ---------------------------TripletAttention Begin---------------------------
class BasicConv_T(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            relu=True,
            bn=True,
            bias=False,
    ):
        super(BasicConv_T, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv_T(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out

# ---------------------------TripletAttention End---------------------------
