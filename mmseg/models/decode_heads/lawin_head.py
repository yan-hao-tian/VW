import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, NonLocal2d, DepthwiseSeparableConvModule


from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class MLP(nn.Module):
    """
    Linear Embedding: github.com/NVlabs/SegFormer
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding: github.com/SwinTransformer/
    """
    def __init__(self, proj_type='pool', patch_size=4, embed_dim=96, norm_cfg=None, act_cfg=None, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj_type = proj_type
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        if proj_type == 'conv':
            self.proj = DepthwiseSeparableConvModule(
                                embed_dim, 
                                embed_dim, 
                                kernel_size=patch_size, 
                                stride=patch_size, 
                                padding=0, 
                                norm_cfg=norm_cfg, 
                                act_cfg=act_cfg)
        elif proj_type == 'pool':
            self.proj = nn.ModuleList([nn.MaxPool2d(kernel_size=patch_size, stride=patch_size), nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)])
        else:
            raise NotImplementedError(f'{proj_type} is not currently supported.')
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        
        if self.proj_type == 'conv': 
            x = self.proj(x)  # B C Wh Ww
        else:
            x = 0.5 * (self.proj[0](x) + self.proj[1](x))
        
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class LawinAttn(NonLocal2d):
    def __init__(self, *arg, head=1,
            patch_size=None, mixing=True, **kwargs):
        super().__init__(*arg, **kwargs)
        self.head = head
        self.patch_size = patch_size
        self.mixing = mixing
        
        if mixing:
            self.norm = nn.LayerNorm(self.in_channels)
            # self.position_mixing = nn.ModuleList([nn.Linear(patch_size*patch_size, patch_size*patch_size) for _ in range(self.head)])
            self.position_mixing = nn.ParameterDict({
                'weight': nn.Parameter(torch.zeros(head, patch_size ** 2, patch_size ** 2)),
                'bias': nn.Parameter(torch.zeros(1, head, 1, patch_size ** 2))
            })
            self.head_mixing = nn.Conv2d(head, head, 1, bias=True)

    def forward(self, query, context):
        # x: [N, C, H, W]
        
        n, c, h, w = context.shape
        
        if self.mixing:
            context = context.reshape(n, c, -1)
            context = self.norm(context.transpose(1, -1))
            context = rearrange(context, 'b n (h d) -> b h d n', h=self.head)
            context_mlp = torch.einsum('bhdn, hnm -> bhdm', context, self.position_mixing['weight']) + self.position_mixing['bias']
            # context_mlp = []
            # for hd in range(self.head):
            #     context_crt = context[:, hd, :, :].unsqueeze(1)
            #     context_mlp.append(self.position_mixing[hd](context_crt))

            # context_mlp = torch.cat(context_mlp, dim=1)
            context_mlp = self.head_mixing(context_mlp)
            context = context+context_mlp
            context = context.reshape(n, c, h, w)

        # g_x: [N, HxW, C]
        g_x = self.g(context).view(n, self.inter_channels, -1)
        g_x = rearrange(g_x, 'b (h dim) n -> (b h) dim n', h=self.head)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = query.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(context).view(n, self.in_channels, -1)
            else:
                phi_x = context.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(query).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(query).view(n, self.inter_channels, -1)
            theta_x = rearrange(theta_x, 'b (h dim) n -> (b h) dim n', h=self.head)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(context).view(n, self.inter_channels, -1)
            phi_x = rearrange(phi_x, 'b (h dim) n -> (b h) dim n', h=self.head)


        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, '(b h) n dim -> b n (h dim)', h=self.head)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *query.size()[2:])

        output = query + self.conv_out(y)

        return output


@HEADS.register_module()
class LAWINHead(BaseDecodeHead):
    def __init__(self, depth, concat_fuse, **kwargs):
        super(LAWINHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.depth = depth
        self.concat_fuse = concat_fuse
        decoder_params = kwargs['decoder_params']

        mixing = decoder_params['mixing']
        in_dim = decoder_params['in_dim']
        embed_dim = decoder_params['embed_dim'] 
        use_scale = decoder_params['use_scale']
        proj_type = decoder_params['proj_type']
        reduction = decoder_params['reduction']
        

        ############### MLP decoder on C1-C4 ###########
        self.linear_c4 = MLP(input_dim=self.in_channels[-1], embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=self.in_channels[2], embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=self.in_channels[1], embed_dim=embed_dim)
        self.linear_fuse = ConvModule(
                                in_channels=embed_dim*3,
                                out_channels=in_dim,
                                kernel_size=1,
                                norm_cfg=dict(type='SyncBN', requires_grad=True)) if concat_fuse else nn.Identity()
        self.linear_pred = nn.Conv2d(in_dim, self.num_classes, kernel_size=1)

        ############# Lawin Transformer ###############
        self.lawin_8 = nn.ModuleList(
                            [LawinAttn(
                                in_channels=in_dim, 
                                reduction=reduction,
                                use_scale=use_scale, 
                                conv_cfg=self.conv_cfg, 
                                norm_cfg=self.norm_cfg, 
                                mode='embedded_gaussian', 
                                head=64, 
                                patch_size=8,
                                mixing=mixing) for _ in range(depth)])
        self.ds_8 = nn.ModuleList(
                            [PatchEmbed(
                                proj_type,
                                patch_size=8,
                                embed_dim=in_dim,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg) for _ in range(depth)])

        self.lawin_4 = nn.ModuleList(
                            [LawinAttn(
                                in_channels=in_dim, 
                                reduction=reduction,
                                use_scale=use_scale, 
                                conv_cfg=self.conv_cfg, 
                                norm_cfg=self.norm_cfg, 
                                mode='embedded_gaussian', 
                                head=16, 
                                patch_size=8,
                                mixing=mixing) for _ in range(depth)])      
        self.ds_4 = nn.ModuleList(
                            [PatchEmbed(
                                proj_type,
                                patch_size=4,
                                embed_dim=in_dim,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg) for _ in range(depth)])

        self.lawin_2 = nn.ModuleList(
                            [LawinAttn(
                                in_channels=in_dim, 
                                reduction=reduction,
                                use_scale=use_scale, 
                                conv_cfg=self.conv_cfg, 
                                norm_cfg=self.norm_cfg, 
                                mode='embedded_gaussian', 
                                head=4, 
                                patch_size=8,
                                mixing=mixing) for _ in range(depth)])   
        self.ds_2 = nn.ModuleList(
                            [PatchEmbed(
                                proj_type,
                                patch_size=2,
                                embed_dim=in_dim,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg) for _ in range(depth)])
        
        ############# Multi-Scale Aggregation ###############
        self.short_path = ConvModule(
                                in_channels=in_dim,
                                out_channels=in_dim,
                                kernel_size=1,
                                norm_cfg=dict(type='SyncBN', requires_grad=True))
        self.image_pool = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1), 
                                ConvModule(in_dim, in_dim, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)) 
        self.cat = ConvModule(
                        in_channels=in_dim*5,
                        out_channels=in_dim,
                        kernel_size=1,
                        norm_cfg=dict(type='SyncBN', requires_grad=True)) if concat_fuse else nn.Identity()

        ############### Low-level feature enhancement ###########
        self.linear_c1 = MLP(input_dim=self.in_channels[0], embed_dim=48)
        self.low_level_fuse = ConvModule(
                                    in_channels=in_dim+48,
                                    out_channels=in_dim,
                                    kernel_size=1,
                                    norm_cfg=dict(type='SyncBN', requires_grad=True))

    def get_query(self, x, patch_size):
        n, _, h, w = x.shape
        ### get query ###
        query = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        query = rearrange(query, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size, pw=patch_size, nh=h//patch_size, nw=w//patch_size)

        return query

    def get_context_fast(self, x, r, patch_size, i):
        _, _, h, w = x.shape
        # padding
        if r in [2, 4]:
            p = [patch_size//2-patch_size//r//2 for _ in range(4)]
        elif r == 8:
            p = [3, 4, 4, 3]
        else:
            raise NotImplementedError(f'ratio {r} is not currently supported.')
        
        context = getattr(self, f'ds_{r}')[i](x)
        context = F.pad(context, p)
        context = F.unfold(context, kernel_size=patch_size, stride=patch_size//r)
        context = rearrange(context, 'b (c ph pw) (nh nw) -> (b nh nw) c ph pw', ph=patch_size, pw=patch_size, nh=h//patch_size, nw=w//patch_size)
        
        return context
    
    def forward(self, inputs):

        inputs = self._transform_inputs(inputs)
        c1, c2, c3, c4 = inputs

        ############### MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) # (n, c, 32, 32)
        _c4 = resize(_c4, size=c2.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c2.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1) if self.concat_fuse else _c4 + _c3 + _c2) #(n, c, 128, 128)
        n, _, h, w = _c.shape

        ############# Multi-Scale Aggregation ###############
        patch_size = 8
        output = []
        output.append(self.short_path(_c))
        output.append(resize(self.image_pool(_c),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=self.align_corners))
        
        for _, r in enumerate([8, 4, 2]):

            for i in range(self.depth):
                feat = _c if i ==0 else _output
                query, context = self.get_query(feat, patch_size), self.get_context_fast(feat, r, patch_size, i)
                _output = getattr(self, f'lawin_{r}')[i](query, context)
                _output = rearrange(_output, '(b nh nw) c ph pw -> b c (nh ph) (nw pw)', ph=patch_size, pw=patch_size, nh=h//patch_size, nw=w//patch_size)

            output.append(_output)
        
        output = self.cat(torch.cat(output, dim=1) if self.concat_fuse else sum(output))

        ############### Low-level feature enhancement ###########
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        output = resize(output, size=c1.size()[2:], mode='bilinear', align_corners=False)
        output = self.low_level_fuse(torch.cat([output, _c1], dim=1))

        output = self.dropout(output)
        output = self.linear_pred(output)

        return output
