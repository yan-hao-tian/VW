from abc import ABCMeta

import torch
import torch.nn as nn
torch.set_printoptions(profile="full")

from ..utils import constant_init, normal_init
from .conv_module import ConvModule
from .context_block import ContextBlock
from .registry import PLUGIN_LAYERS
from einops import rearrange

import torch.nn.functional as F
from mmseg.ops import resize
import numpy as np
np.set_printoptions(threshold=np.inf)
class _NonLocalNd(nn.Module, metaclass=ABCMeta):
    """Basic Non-local module.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971
    Code reference: https://github.com/AlexHex7/Non-local_pytorch

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
            Default: True.
        conv_cfg (None | dict): The config dict for convolution layers.
            If not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (None | dict): The config dict for normalization layers.
            Default: None. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
    """

    def __init__(self,
            in_channels,
            reduction=2,
            use_scale=True,
            conv_cfg=None,
            norm_cfg=None,
            mode='embedded_gaussian',
            head=1,
            patch_size=None,
            **kwargs):
        super(_NonLocalNd, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode
        self.head = head
        self.flag = 0

        #---------
        # mlp
        #---------
        if self.head!=1:
            for hd in range(1, self.head+1):
                setattr(self, f'mlp_{hd}', nn.Linear(patch_size*patch_size, patch_size*patch_size))
            
        #--------
        # rela PE
        #---------
        #num_attention_heads=1 # attention head
        #max_position_embeddings = 100 # Max length of sequence
        #position_embedding_size = self.inter_channels // 2// head
        #self.row_embeddings = nn.Embedding(2 * max_position_embeddings - 1, position_embedding_size)
        #self.col_embeddings = nn.Embedding(2 * max_position_embeddings - 1, position_embedding_size)
        # self.head_keys_row = nn.Linear(position_embedding_size, num_attention_heads, bias=False)
        # self.head_keys_col = nn.Linear(position_embedding_size, num_attention_heads, bias=False)
        #deltas = torch.arange(max_position_embeddings).view(1, -1) - torch.arange(max_position_embeddings).view(-1, 1)
        #relative_indices = deltas + max_position_embeddings - 1
        #self.register_buffer("relative_indices", relative_indices)

        if mode not in [
                'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
        ]:
            raise ValueError("Mode should be in 'gaussian', 'concatenation', "
                             f"'embedded_gaussian' or 'dot_product', but got "
                             f'{mode} instead.')

        # g, theta, phi are defaulted as `nn.ConvNd`.
        # Here we use ConvModule for potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)
        # NOTE
        self.conv_out = ConvModule(
            256,
            512,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.drop_path = DropPath(drop_prob=0)
        
        if self.mode != 'gaussian':
            self.theta = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None)
            self.phi = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None)

        if self.mode == 'concatenation':
            self.concat_project = ConvModule(
                self.inter_channels * 2,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act_cfg=dict(type='ReLU'))

        # NOTE: nonlocal meets gc
        # self.context_k = ContextBlock(in_channels=self.inter_channels, ratio=1)
        # self.context_v = ContextBlock(in_channels=self.inter_channels, ratio=1)
        

        self.init_weights(**kwargs)

    def init_weights(self, std=0.01, zeros_init=True):
        if self.mode != 'gaussian':
            for m in [self.g, self.theta, self.phi]:
                normal_init(m.conv, std=std)
        else:
            normal_init(self.g.conv, std=std)
        if zeros_init:
            if self.conv_out.norm_cfg is None:
                constant_init(self.conv_out.conv, 0)
            else:
                constant_init(self.conv_out.norm, 0)
        else:
            if self.conv_out.norm_cfg is None:
                normal_init(self.conv_out.conv, std=std)
            else:
                normal_init(self.conv_out.norm, std=std)
        if self.head!=1:
            for hd in range(1, self.head+1):
                constant_init(getattr(self, f'mlp_{hd}'), val=1, bias=0)

    def gaussian(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x, phi_x, attention_scores):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is 'self.inter_channels//head'
            attention_scores /= theta_x.shape[-1]**0.5
            pairwise_weight /= theta_x.shape[-1]**0.5
        
        #with open('/home/yanhaotian/yht/mmsegmentation/attn_infer.txt', 'a') as f:
        #    f.write(str(pairwise_weight))
        #    f.write('\n')
        
        pairwise_weight += attention_scores
        
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        ##pairwise_weight_ = pairwise_weight.reshape(pairwise_weight.size(0)//16, 16,-1)[0, :, :]
        # if self.flag<3:
        #    with open('attn_infer.txt', 'a') as f:
        #        f.write(str(pairwise_weight_.cpu().detach().numpy()))
        #        f.write('\n')
        #    self.flag+=1
            
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x, phi_x):
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]

        return pairwise_weight

    def forward(self, x, output_attn):
        # Assume `reduction = 1`, then `inter_channels = C`
        # or `inter_channels = C` when `mode="gaussian"`
        # NonLocal1d x: [N, C, H]
        # NonLocal2d x: [N, C, H, W]
        # NonLocal3d x: [N, C, T, H, W]
        head = self.head
        n, c, h, w = x.shape
        # -------
        # rela pe
        # -------
        #context_h = x.size(2)
        #query_h = output_attn.size(2)
        #context_w = x.size(3)
        #query_w = output_attn.size(3)

        #----------------
        # mlp on input X
        #----------------
        if self.head!=1:
            x = x.reshape(n, c, -1)
            x_mlp = []
            for hd in range(1, self.head+1):
                x_mlp.append(getattr(self, f'mlp_{hd}')(x[:, (hd-1)//self.head*c:hd//self.head*c, :]))
            
            x_mlp = torch.cat(x_mlp, dim=1)
            x = x+x_mlp
            x = x.reshape(n, c, h, w)
        # NOTE
        # patch_size = output_attn.size(2)
        # x = adaptive_context_vector(x, patch_size=patch_size) # (bs, c, n)
        # n = x.size(0)

        # NonLocal1d g_x: [N, H, C]
        # NonLocal2d g_x: [N, HxW, C]
        # NonLocal3d g_x: [N, TxHxW, C]
        g_x = self.g(x)
        #g_x, g_x_aux = g_x[:, 0:192, :, :]   ,g_x[:, 192:, :, :]
        # g_x = self.g(torch.nn.functional.adaptive_avg_pool2d(x, (h//2, w//2)))
        # NOTE: The nonlocal meets gc block
       
        #g_x = g_x.reshape(n, self.inter_channels, -1)
        #g_x_1 = self.mlp_1_v(g_x[:, :self.inter_channels//4, :])
        #g_x_2 = self.mlp_2_v(g_x[:, self.inter_channels//4:self.inter_channels//2, :])
        #g_x_3 = self.mlp_3_v(g_x[:, self.inter_channels//2:self.inter_channels//4*3, :])
        #g_x_4 = self.mlp_4_v(g_x[:, self.inter_channels//4*3:, :])
        #g_x += torch.cat([g_x_1, g_x_2, g_x_3, g_x_4], dim=1)
        #g_x = g_x.reshape(n,self.inter_channels, h, w)
        
        #g_x = F.unfold(g_x, kernel_size=(2, 2), stride=2) # unfold for gc
        #g_x = rearrange(g_x, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=2, psw=2, pnh=h//2, pnw=w//2) # rearrange for gc (b, c, 2, 2)
        #g_x = self.context_v(g_x) # (b, c, 1, 1)
        #g_x = rearrange(g_x, ' (b pnh pnw) c psh psw -> b (c psh psw) pnh pnw', psh=1, psw=1, pnh=h//2, pnw=w//2) # rearrange for nonlocal(b, c, 32, 64)
        #g_x = F.adaptive_avg_pool2d(g_x, (h//2, w//2))
        #g_x = F.unfold(g_x, kernel_size=(int(patch_size), int(patch_size)), stride=patch_size//2, padding=patch_size//4)
        #g_x = rearrange(g_x, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=int(patch_size), psw=int(patch_size), pnh=patch_num_h, pnw=patch_num_w)
        n = g_x.size(0)
        g_x = g_x.view(n, self.inter_channels, -1)

        # NOTE
        g_x = rearrange(g_x, 'b (h dim) n -> (b h) dim n', h=head)
        g_x = g_x.permute(0, 2, 1)

        #---------
        # rela pe
        #---------
        #relative_indices = self.relative_indices[:query_w,:context_w].reshape(-1)
        #row_embeddings = self.row_embeddings(relative_indices)
        #relative_indices = self.relative_indices[:query_h,:context_h].reshape(-1)
        #col_embeddings = self.col_embeddings(relative_indices)


        # NonLocal1d theta_x: [N, H, C], phi_x: [N, C, H]
        # NonLocal2d theta_x: [N, HxW, C], phiasdi_x: [N, C, HxW]
        # NonLocal3d theta_x: [N, TxHxW, C], phi_x: [N, C, TxHxW]
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            
            #theta_x = F.unfold(output_attn, kernel_size=(int(patch_size), int(patch_size)), stride=patch_size)
            #theta_x = rearrange(theta_x, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=int(patch_size), psw=int(patch_size), pnh=patch_num_h, pnw=patch_num_w)
            #memory = theta_x
            theta_x = self.theta(output_attn)
            #theta_x, theta_x_aux = theta_x[:,0:192,:, :], theta_x[:, 192:, :, :]
            n = theta_x.size(0)
            theta_x = theta_x.view(n, self.inter_channels, -1)
            # NOTE: multihead
            theta_x = rearrange(theta_x, 'b (h dim) n -> (b h) dim n', h=head)

            #---------
            # rela pe
            #---------
            #theta_x_row = theta_x.reshape(-1, self.inter_channels//head, query_h, query_w)[:, :self.inter_channels // 2 //head, :, :].permute(0, 3, 2, 1)
            #theta_x_col = theta_x.reshape(-1, self.inter_channels//head, query_h, query_w)[:, self.inter_channels // 2//head:, :, :].permute(0, 3, 2, 1)
            #row_scores = torch.einsum("bijd,ikd->bijk", theta_x_row, row_embeddings.view(query_w, context_w, -1))
            #col_scores = torch.einsum("bijd,jld->bijl", theta_x_col, col_embeddings.view(query_h, context_h, -1))
            #attention_scores = row_scores.unsqueeze(-1) + col_scores.unsqueeze(-2)
            ## attention_scores = attention_scores.permute(0, 2, 1, 4, 3) / (self.inter_channels ** 0.5) # b, h, w, d, h, w
            #attention_scores = attention_scores.reshape(-1, query_h*query_w, context_h*context_w)

            theta_x = theta_x.permute(0, 2, 1)
            

            # phi_x = self.phi(torch.nn.functional.adaptive_avg_pool2d(x, (h//2, w//2)))
            phi_x = self.phi(x)
            # phi_x, phi_x_aux = phi_x[:, 0:192, :, :], phi_x[:, 192:, :, :]
            # NOTE: nonlocal meet gc
            #phi_x = F.unfold(phi_x, kernel_size=(2, 2), stride=2) # unfold for gc
            #phi_x = rearrange(phi_x, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=2, psw=2, pnh=h//2, pnw=w//2) # rearrange for gc (b, c, 2, 2)
            #phi_x = self.context_k(phi_x) # (b, c, 1, 1)
            #phi_x = rearrange(phi_x, ' (b pnh pnw) c psh psw -> b (c psh psw) pnh pnw', psh=1, psw=1, pnh=h//2, pnw=w//2) # rearrange for nonlocal(b, c, 32, 64)
            #phi_x = F.adaptive_avg_pool2d(phi_x, (h//2, w//2))
            #phi_x = F.unfold(phi_x, kernel_size=(int(patch_size), int(patch_size)), stride=patch_size//2, padding=patch_size//4)
            #phi_x = rearrange(phi_x, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=int(patch_size), psw=int(patch_size), pnh=patch_num_h, pnw=patch_num_w)
            n = phi_x.size(0)
            
            #phi_x = phi_x.reshape(n, self.inter_channels, -1)
            #phi_x_1 = self.mlp_1_k(phi_x[:, :self.inter_channels//4, :])
            #phi_x_2 = self.mlp_2_k(phi_x[:, self.inter_channels//4:self.inter_channels//2, :])
            #phi_x_3 = self.mlp_3_k(phi_x[:, self.inter_channels//2:self.inter_channels//4*3, :])
            #phi_x_4 = self.mlp_4_k(phi_x[:, self.inter_channels//4*3:, :])
            #phi_x += torch.cat([phi_x_1, phi_x_2, phi_x_3, phi_x_4], dim=1)
            #phi_x = phi_x.reshape(n,self.inter_channels, h, w)

            phi_x = phi_x.view(n, self.inter_channels, -1)

            

            # NOTE
            phi_x = rearrange(phi_x, 'b (h dim) n -> (b h) dim n', h=head)

        pairwise_func = getattr(self, self.mode)
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x, attention_scores=0)
        
        # NonLocal1d y: [N, H, C]
        # NonLocal2d y: [N, HxW, C]
        # NonLocal3d y: [N, TxHxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # NonLocal1d y: [N, C, H]
        # NonLocal2d y: [N, C, H, W]
        # NonLocal3d y: [N, C, T, H, W]
        #NOTE:
        y = rearrange(y, '(b h) n dim -> b n (h dim)', h=head)

        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *output_attn.size()[2:])
        # print(y.shape, memory.shape)

        # -----
        # aux attention map
        # -----
#
#        phi_x = F.unfold(phi_x_aux, kernel_size=(int(patch_size*2), int(patch_size*2)), stride=patch_size, padding=patch_size//2)
#        phi_x = rearrange(phi_x, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=int(patch_size*2), psw=int(patch_size*2), pnh=patch_num_h, pnw=patch_num_w)
#        n = phi_x.size(0)
#        phi_x = phi_x.view(n, self.inter_channels//4, -1)
#        pairwise_weight = pairwise_func(theta_x, phi_x)
#
#        g_x = F.unfold(g_x_aux, kernel_size=(int(patch_size*2), int(patch_size*2)), stride=patch_size, padding=patch_size//2)
#        g_x = rearrange(g_x, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=int(patch_size*2), psw=int(patch_size*2), pnh=patch_num_h, pnw=patch_num_w)
#        n = g_x.size(0)
#        g_x = g_x.view(n, self.inter_channels//4, -1)
#        g_x = g_x.permute(0, 2, 1)
##         print(g_x.shape, pairwise_weight.shape)
#        y_ = torch.matmul(pairwise_weight, g_x)
#        
#        y_ = y_.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels//4, patch_size//2, patch_size//2)
#        y_ = resize(y_, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
#        y = torch.cat([y, y_], dim=1)

        output = output_attn + self.drop_path(self.conv_out(y))
        # NOTE: no shortcut
        # output = self.drop_path(self.conv_out(y))

        return output


class NonLocal1d(_NonLocalNd):
    """1D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv1d').
    """

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 conv_cfg=dict(type='Conv1d'),
                 **kwargs):
        super(NonLocal1d, self).__init__(
            in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


@PLUGIN_LAYERS.register_module()
class NonLocal2d(_NonLocalNd):
    """2D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv2d').
    """

    _abbr_ = 'nonlocal_block'

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 conv_cfg=dict(type='Conv2d'),
                 **kwargs):
        super(NonLocal2d, self).__init__(
            in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class NonLocal3d(_NonLocalNd):
    """3D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv3d').
    """

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 conv_cfg=dict(type='Conv3d'),
                 **kwargs):
        super(NonLocal3d, self).__init__(
            in_channels, conv_cfg=conv_cfg, **kwargs)
        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def adaptive_context_vector(x, patch_size):
    # input_x: n, c, h, w
    bs, c, h, w = x.shape
    pad = patch_size 
    kernel_size = patch_size * 3
    stride = patch_size
    patch_num_h = h//patch_size
    patch_num_w = w//patch_size
    # x_c_size = patch_size*3//8
    x_c_size = patch_size*4//8
    x_d_size = patch_size*2//8
    x_e_size = patch_size*1//8

    # print(x_c_size, patch_num_w, patch_num_h)

    bs = patch_num_h*patch_num_w*bs

    x = F.unfold(x, kernel_size=kernel_size, padding=pad, stride=stride)
    x = rearrange(x, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw ', psh=kernel_size, psw=kernel_size, pnh=patch_num_h, pnw=patch_num_w)
    x_c = F.adaptive_avg_pool2d(x[:, :, patch_size:-patch_size, patch_size:-patch_size], output_size=(2*x_c_size, 2*x_c_size)).view(bs, c, -1) # (bs, c, n)
    x_d = torch.cat([
        F.adaptive_avg_pool2d(x[:, :, patch_size//2:patch_size, patch_size//2:-patch_size//2], output_size=(x_d_size, 4*x_d_size)).view(bs, c, -1),
        F.adaptive_avg_pool2d(x[:, :, -patch_size:-patch_size//2, patch_size//2:-patch_size//2], output_size=(x_d_size, 4*x_d_size)).view(bs, c, -1),
        F.adaptive_avg_pool2d(x[:, :, patch_size:-patch_size, patch_size//2:patch_size], output_size=(2*x_d_size, x_d_size)).view(bs, c, -1),
        F.adaptive_avg_pool2d(x[:, :, patch_size:-patch_size, -patch_size:-patch_size//2], output_size=(2*x_d_size, x_d_size)).view(bs, c, -1)
    ], dim=-1)
    x_e = torch.cat([
        F.adaptive_avg_pool2d(x[:, :, :patch_size//2,:], output_size=(x_e_size, 6*x_e_size)).view(bs, c, -1),
        F.adaptive_avg_pool2d(x[:, :, -patch_size//2:, :], output_size=(x_e_size, 6*x_e_size)).view(bs, c, -1),
        F.adaptive_avg_pool2d(x[:, :, patch_size//2:-patch_size//2, :patch_size//2], output_size=(4*x_e_size, x_e_size)).view(bs, c, -1),
        F.adaptive_avg_pool2d(x[:, :, patch_size//2:-patch_size//2, -patch_size//2:], output_size=(4*x_e_size, x_e_size)).view(bs, c, -1)
    ], dim=-1)

    return torch.cat([x_c, x_d, x_e], dim=-1)
