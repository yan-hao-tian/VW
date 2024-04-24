import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, NonLocal2d
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from einops import rearrange
import math
from mmseg.ops import resize
from functools import partial

class VWA(NonLocal2d):
    def __init__(self, *arg, in_channels_m=512, head=1, **kwargs):
        super().__init__(*arg, **kwargs)
        self.head = head
        
        self.g = ConvModule(
            in_channels_m,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=dict(type='Conv2d'),
            act_cfg=None)
        
        self.phi = ConvModule(
            in_channels_m,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=dict(type='Conv2d'),
            act_cfg=None)
      
    def forward(self, query, context_k, context_v):
        n, c, h, w = context_v.shape

        # g_x: [N, HxW, C]
        g_x = self.g(context_v).view(n, self.inter_channels, -1)
        g_x = rearrange(g_x, 'b (h dim) n -> (b h) dim n', h=self.head)
        g_x = g_x.permute(0, 2, 1)
       
        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = query.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(context_k).view(n, self.in_channels, -1)
            else:
                phi_x = context_k.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(query).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(context_k).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(query).view(n, self.inter_channels, -1)
            theta_x = rearrange(theta_x, 'b (h dim) n -> (b h) dim n', h=self.head)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(context_k).view(n, self.inter_channels, -1)
            phi_x = rearrange(phi_x, 'b (h dim) n -> (b h) dim n', h=self.head)
           
        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x) # TODO: attention map
        
        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, '(b h) n dim -> b n (h dim)', h=self.head)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *query.size()[2:])
      
        output = self.conv_out(y)

        return output

class MLP(nn.Module):
    """
    Linear Embedding: github.com/NVlabs/SegFormer
    """
    def __init__(self, input_dim=2048, embed_dim=768, identity=False):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        if identity:
            self.proj = nn.Identity()

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.permute(0,2,1).reshape(n, -1, h, w)
        
        return x
    
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

@HEADS.register_module()
class VWHead(BaseDecodeHead):
    def __init__(self, short_cut, nheads, **kwargs):
        super(VWHead, self).__init__(input_transform='multiple_select', **kwargs)
        embed_dim = self.channels
        self.kernel = kernel = [[2, 2], [4, 4], [8, 8]]
        self.pre_scaling = True
        self.short_cut = short_cut
        self.linear_c4 = MLP(self.in_channels[-1], 512)
        self.linear_c3 = MLP(self.in_channels[2], 512)
        self.linear_c2 = MLP(self.in_channels[1], 512)
       
        self.linear_fuse = ConvModule(
                                in_channels=embed_dim*3,
                                out_channels=embed_dim,
                                kernel_size=1,
                                norm_cfg=dict(type='SyncBN', requires_grad=True))
        
        self.attn = nn.Sequential(*[VWA(in_channels=embed_dim, 
                                            reduction=1,
                                            use_scale=True, 
                                            conv_cfg=self.conv_cfg, 
                                            norm_cfg=self.norm_cfg, 
                                            mode='embedded_gaussian',
                                            in_channels_m=embed_dim,
                                            head=nheads)
                                                    for _ in kernel])  
        self.ds = nn.ModuleList([ConvModule(
                                in_channels=embed_dim,
                                out_channels=embed_dim//(k1*k2),
                                kernel_size=(k1, k2),
                                stride=(1, 1),
                                norm_cfg=dict(type='SyncBN', requires_grad=True)) 
                                        for k1, k2 in kernel])
        
        
        self.short_path = ConvModule(
                            in_channels=embed_dim,
                            out_channels=embed_dim,
                            kernel_size=1,
                            norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.image_pool = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1), 
                                ConvModule(embed_dim, embed_dim, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        self.cat = ConvModule(in_channels=embed_dim*5,
                                out_channels=embed_dim,
                                kernel_size=1,
                                norm_cfg=dict(type='SyncBN', requires_grad=True)) 
        
        ############### Low-level feature enhancement ###########
        self.linear_c1 = MLP(input_dim=self.in_channels[0], embed_dim=48)
        self.low_level_fuse = ConvModule(
                                    in_channels=embed_dim+48,
                                    out_channels=embed_dim,
                                    kernel_size=1,
                                    padding=0,
                                    norm_cfg=dict(type='SyncBN', requires_grad=True))
        
        
    def mlp_decoder(self, inputs):
        c1, c2, c3, c4 = inputs

        _c4 = self.linear_c4(c4) # (n, c, 32, 32)
        _c3 = self.linear_c3(c3)
        _c2 = self.linear_c2(c2)
   
        _c4 = resize(_c4, size=inputs[1].size()[2:],mode='bilinear',align_corners=False)
        _c3 = resize(_c3, size=inputs[1].size()[2:],mode='bilinear',align_corners=False)
        # _c2 = resize(_c2, size=inputs[stride-1].size()[2:],mode='bilinear',align_corners=False)
       
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1)) #(n, c, 128, 128)
        
        return _c, c1

    def copy_shift_padding(self, context, rh, rw, nh, nw, ph, pw):
        n, _, h, w = context.shape
        if rw > nw:
            pad_w_ = rw*pw - w
            context = F.pad(context, (pad_w_//2, pad_w_-pad_w_//2, 0, 0))
            context = torch.cat([context[...,(rw+1)*pw//2:rw*pw],
                                context[...,pad_w_//2:pad_w_//2+w],
                                context[...,-(rw*pw):-((rw+1)*pw//2)]
                                ], dim=3)
        else:
            context = torch.cat([context[...,(rw+1)*pw//2:rw*pw],
                                context,
                                context[...,-(rw*pw):-((rw+1)*pw//2)]
                                ], dim=3)
        if rh > nh:
            pad_h_ = rh*ph - h
            context = F.pad(context, (0, 0, pad_h_//2, pad_h_-pad_h_//2))
            context = torch.cat([context[...,(rh+1)*ph//2:rh*ph,:],
                                context[...,pad_h_//2:pad_h_//2+h,:],
                                context[...,-(rh*ph):-((rh+1)*ph//2),:]
                                ], dim=2) 
        else:
            context = torch.cat([context[...,(rh+1)*ph//2:rh*ph,:],
                                context,
                                context[...,-(rh*ph):-((rh+1)*ph//2),:]
                                ], dim=2) 
           
        return context
    
    def varying_context_window(self, context, rh, rw, nh, nw, ph, pw):
        context = self.copy_shift_padding(context, rh, rw, nh, nw, ph, pw)
        context = F.unfold(context, 
                           kernel_size=(rh*ph, rw*pw), 
                           stride=(ph, pw), 
                           )
        context = rearrange(context, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', 
                            ph=ph*rh, pw=pw*rw, nh=nh, nw=nw)
        return context
           
    def dope_scaling(self, context, rh, rw, j):
        pad_w = [rw//2-1, rw//2] if rw%2 ==0 else [rw//2]*2
        pad_h = [rh//2-1, rh//2] if rh%2 ==0 else [rh//2]*2
        context = F.pad(context, pad_w+pad_h)
        context = getattr(self, f'ds')[j](context)   
        return context
    
    def vwformer(self, _c):
        output = [self.short_path(_c), 
                  resize(self.image_pool(_c),
                        size=_c.size()[2:],
                        mode='bilinear',
                        align_corners=self.align_corners)]
        
        _, _, h, w = _c.shape
        nh, nw = 8, 8
        ph, pw = h // nh, w // nw
        # ph, pw = 10, 10
        # nh, nw = h // ph, w // pw
       
        for j, r in enumerate(self.kernel):
            rh, rw = r
            # if r !=8 : 
                
            query = rearrange(_c, 'b c (nh ph) (nw pw) -> (b nh nw) c ph pw', nh=nh, nw=nw)
            
            if self.pre_scaling:
                context = self.dope_scaling(_c, rh, rw, j)
                context = self.varying_context_window(context, rh, rw, nh, nw, ph, pw) 
            else:
                context = self.varying_context_window(_c, rh, rw, nh, nw, ph, pw)
                context = self.dope_scaling(context, rh, rw, j)
            
            context = rearrange(context, 'b c (ph rh) (pw rw) -> b (rh rw c) ph pw', rh=rh, rw=rw)
        
            _output = getattr(self, f'attn')[j](query, context, context)
            _output = rearrange(_output, '(b nh nw) c ph pw -> b c (nh ph) (nw pw)', nh=nh, nw=nw)
            
            if self.short_cut: _output += _c
            # else:
            #     query = _c
            #     context = self.varying_context_window(_c, rh, rw, nh, nw, ph, pw)
            #     _output = getattr(self, f'attn')[j](query, context, context)
            
            output += [_output]  
                
        
        output = self.cat(torch.cat(output, dim=1))
        return output
    
    def low_level(self, _c, c1):
        
        _c1 = self.linear_c1(c1)
        output = resize(_c, 
                        size=c1.size()[2:], 
                        mode='bilinear', 
                        align_corners=False)
        output = self.low_level_fuse(torch.cat([output, _c1], dim=1))
        
        return output
    
    def forward(self, inputs):
        _c, c1 = self.mlp_decoder(inputs)        
 
        _c = self.vwformer(_c)
        # _c = self.vwformer(_c)
        _c = self.low_level(_c, c1)

        output = self.cls_seg(_c)
    
        return output
    
@HEADS.register_module()
class VWCityHead(BaseDecodeHead): # vw for cityscapes
    def __init__(self, short_cut, nheads, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        self.short_cut = short_cut
        self.feature_channels = feature_channels = self.in_channels

        Conv2d = partial(ConvModule, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        
        embed_dim = self.channels
        
        self.linear_c4 = Conv2d(in_channels=feature_channels[3], out_channels=embed_dim, kernel_size=1)
        self.linear_c3 = Conv2d(in_channels=feature_channels[2], out_channels=embed_dim, kernel_size=1)
        self.linear_c2 = Conv2d(in_channels=feature_channels[1], out_channels=embed_dim, kernel_size=1)
        self.linear_c1 = Conv2d(in_channels=feature_channels[0], out_channels=48, kernel_size=1)
        self.linear_fuse = Conv2d(in_channels=embed_dim*3, out_channels=embed_dim, kernel_size=1)
        self.short_path = Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)     
        # self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
        #                     Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1))
        
        self.cat = Conv2d(in_channels=embed_dim*4, out_channels=embed_dim, kernel_size=1)
        self.low_level_fuse = Conv2d(in_channels=embed_dim+48, out_channels=embed_dim, kernel_size=1)
        self.attn = nn.Sequential(*[nn.MultiheadAttention(embed_dim, nheads, batch_first=False) for _ in [2, 4, 8]]) 
        # self.attn = nn.Sequential(*[MultiheadNonLocal(in_channels=embed_dim, norm_cfg=self.norm_cfg, head=nheads) for _ in [2, 4, 8]]) 
        
        self.ds = nn.ModuleList([Conv2d(in_channels=embed_dim, out_channels=embed_dim//(k*k), kernel_size=k, padding=0)
                                    for k in [2, 4, 8]]) 


    def forward_features(self, inputs):
        
        c4, c3, c2, c1 = inputs[::-1]
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=c2.size()[2:],mode='nearest')
        
        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=c2.size()[2:],mode='nearest')
        
        _c2 = self.linear_c2(c2)
       
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1)) #(n, c, 128, 128)
        n, _, h, w = _c.shape
        
        output = []
        output.append(self.short_path(_c))
        
        _, _, h, w = _c.shape
        nh, nw = 8, 8
        ph, pw = h // nh, w // nw
        query = rearrange(_c, 'b c (nh ph) (nw pw) -> (ph pw) (b nh nw) c', nh=nh, nw=nw)
        
        
        for j, r in enumerate([2, 4, 8]):
            # if r != 8:
            rh = rw = r
            pad_w = [rw//2-1, rw//2] if rw%2 ==0 else [rw//2]*4
            pad_h = [rh//2-1, rh//2] if rh%2 ==0 else [rh//2]*4
            context = F.pad(_c, pad_w+pad_h)
            context = self.ds[j](context)
           
            context = torch.cat([context[...,(rw+1)*pw//2:rw*pw],
                                context,
                                context[...,-(rw*pw):-((rw+1)*pw//2)]
                                ], dim=3)  
            context = torch.cat([context[...,(rh+1)*ph//2:rh*ph,:],
                                context,
                                context[...,-(rh*ph):-((rh+1)*ph//2),:]
                                ], dim=2) 
            
            context = F.unfold(context, kernel_size=(ph*rh, pw*rw), 
                            stride=(ph, pw))
            context = rearrange(context, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', 
                                ph=ph*rh, pw=pw*rw, nh=nh, nw=nw)
            context = rearrange(context, 'b c (ph nh) (pw nw) -> (ph pw) b (nh nw c)', nh=rh, nw=rw)
            _output, _ = self.attn[j](query, context, context)
            _output = rearrange(_output, '(ph pw) (b nh nw) c -> b c (nh ph) (nw pw)', nh=nh, nw=nw, ph=ph, pw=pw)
            output.append(_output)  
            if self.short_cut: output[-1] += _c
            
        output = self.cat(torch.cat(output, dim=1))
        _c1 = self.linear_c1(c1)
        output = F.interpolate(output, size=c1.size()[2:], mode='bilinear')
        output = self.low_level_fuse(torch.cat([output, _c1], dim=1))
        
        return output

    def forward(self, features, targets=None):
        logits = self.cls_seg(self.forward_features(features))
        return logits