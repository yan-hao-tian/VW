# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, DeformConv, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .msdeformattn import MSDeformAttnTransformerEncoderOnly
from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import TransformerEncoder, TransformerEncoderLayer, _get_clones, _get_activation_fn

from einops import rearrange

def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


# This is a modified FPN decoder.
@SEM_SEG_HEADS_REGISTRY.register()
class BasePixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        depth=1
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]
        self.feature_channels = feature_channels
        self.depth = depth
        
        output_norm = get_norm(norm, 512)
        use_bias = norm == ""
        # self.linear_c5 = Conv2d(feature_channels[-1], 512, kernel_size=3, stride=2, padding=1, bias=use_bias, norm=output_norm, activation=F.relu,)
        self.linear_c4 = Conv2d(feature_channels[-1], 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,)
        self.linear_c3 = Conv2d(feature_channels[-2], 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,)
        self.linear_c2 = Conv2d(feature_channels[1], 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,)
        self.linear_c1 = Conv2d(feature_channels[0], 48, kernel_size=1, bias=use_bias, norm=get_norm('BN', 48), activation=F.relu,)
        self.linear_fuse = Conv2d(512*3, 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,)
        
        self.low_level_fuse = Conv2d(560, 256, kernel_size=1, bias=use_bias, norm=get_norm(norm, 256), activation=F.relu,)
        
        self.proj = nn.ModuleList([Conv2d(feature_channels[i], 256, kernel_size=1, bias=use_bias, norm=get_norm(norm, 256), activation=F.relu,) for i in range(1, 4)])
        
        self.attn_0 = nn.Sequential(*[nn.MultiheadAttention(512, 16, batch_first=True) for _ in [2, 4, 8]])
        self.ds_0 = nn.ModuleList([Conv2d(512, 512//(k**2), kernel_size=k, padding=0,norm=get_norm('BN', 512//(k**2)), activation=F.relu,) 
                                    for k in [2, 4, 8]])
        self.cat_0 = Conv2d(2560, 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,)
        self.short_path_0 = Conv2d(512, 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,)
        self.image_pool_0 = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                            Conv2d(512, 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,))
        
       
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        return ret

    def forward_features(self, features):
        feats = []
        for f in self.in_features[::-1]:
            feats.append(features[f])
        c4, c3, c2, c1 = feats
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=c2.size()[2:],mode='nearest')
        
        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=c2.size()[2:],mode='nearest')
        
        _c2 = self.linear_c2(c2)
       
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1)) #(n, c, 128, 128)
        
        _, _, h, w = _c.shape
        
        output = []
        output.append(self.short_path_0(_c))
        output.append(F.interpolate(
                self.image_pool_0(_c),
                size=_c.size()[2:],
                mode='nearest'))

        nh, nw = 8, 8 
        ph, pw = h // nh, w // nw
        
        query = rearrange(_c, 'b c (nh ph) (nw pw) -> (b nh nw) (ph pw) c', nh=nh, nw=nw)

        for j, r in enumerate([2, 4, 8]):
            rh = rw = r
            pad_w = [rw//2-1, rw//2] if rw%2 ==0 else [rw//2]*4
            pad_h = [rh//2-1, rh//2] if rh%2 ==0 else [rh//2]*4
            context = F.pad(_c, pad_w+pad_h)
            context = self.ds_0[j](context)
        
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
            context = rearrange(context, 'b c (ph nh) (pw nw) -> b (ph pw) (nh nw c)', nh=rh, nw=rw)
            _output, _ = self.attn_0[j](query, context, context)
            _output = rearrange(_output, '(b nh nw) (ph pw) c -> b c (nh ph) (nw pw)', nh=nh, nw=nw, ph=ph, pw=pw)
            
            output.append(_output)  
            
        output = self.cat_0(torch.cat(output, dim=1))

        _c1 = self.linear_c1(c1)
        output = F.interpolate(output, size=c1.size()[2:], mode='bilinear')
        output = self.low_level_fuse(torch.cat([output, _c1], dim=1))
        return output, None, [self.proj[-1](c4), self.proj[-2](c3), self.proj[-3](c2)]
    
    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)


class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


# This is a modified FPN decoder with extra Transformer encoder that processes the lowest-resolution feature map.
@SEM_SEG_HEADS_REGISTRY.register()
class TransformerEncoderPixelDecoder(BasePixelDecoder):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        transformer_pre_norm: bool,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        in_channels = feature_channels[len(self.in_features) - 1]
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)
        self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # update layer
        use_bias = norm == ""
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(output_conv)
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        return ret

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                transformer = self.input_proj(x)
                pos = self.pe_layer(x)
                transformer = self.transformer(transformer, None, pos)
                y = output_conv(transformer)
                # save intermediate feature as input to Transformer decoder
                transformer_encoder_features = transformer
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return self.mask_features(y), transformer_encoder_features, multi_scale_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)

@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        # self.mask_features = Conv2d(
        #     conv_dim,
        #     mask_dim,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        # )
        # weight_init.c2_xavier_fill(self.mask_features)
        
        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        # lateral_convs = []
        # output_convs = []
        depth=1
        use_bias = norm == ""
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]
        self.feature_channels = feature_channels
        self.depth = depth
        output_norm = get_norm(norm, 512)
        use_bias = norm == ""
        self.linear_c4 = Conv2d(256, 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,)
        self.linear_c3 = Conv2d(256, 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,)
        self.linear_c2 = Conv2d(256, 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,)
        self.linear_c1 = Conv2d(feature_channels[0], 48, kernel_size=1, bias=use_bias, norm=get_norm('SyncBN', 48), activation=F.relu,)
        self.linear_fuse = Conv2d(512*3, 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,)
        self.low_level_fuse = Conv2d(560, 256, kernel_size=1, bias=use_bias, norm=get_norm(norm, 256), activation=F.relu,)
        # self.proj = nn.ModuleList([Conv2d(feature_channels[i], 256, kernel_size=1, bias=use_bias, norm=get_norm(norm, 256), activation=F.relu,) for i in range(1, 4)])
        
        for d in range(depth):
            setattr(self, f'attn_{d}', nn.Sequential(*[nn.MultiheadAttention(512, 16, batch_first=True) for _ in [2, 4, 8]]))
        for d in range(depth):
            setattr(self, f'ds_{d}', nn.ModuleList([Conv2d(512, 512//(k**2), kernel_size=k, padding=0,norm=get_norm('SyncBN', 512//(k**2)), activation=F.relu,) 
                                    for k in [2, 4, 8]]))
        for d in range(depth):
            setattr(self, f'cat_{d}', Conv2d(2560, 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,))
        for d in range(depth):
            setattr(self, f'short_path_{d}', Conv2d(512, 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,))
        for d in range(depth):
            setattr(self, f'image_pool_{d}', nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                            Conv2d(512, 512, kernel_size=1, bias=use_bias, norm=output_norm, activation=F.relu,)))
        
        # for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
        #     lateral_norm = get_norm(norm, conv_dim)
        #     output_norm = get_norm(norm, conv_dim)

        #     lateral_conv = Conv2d(
        #         in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
        #     )
        #     output_conv = Conv2d(
        #         conv_dim,
        #         conv_dim,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         bias=use_bias,
        #         norm=output_norm,
        #         activation=F.relu,
        #     )
        #     weight_init.c2_xavier_fill(lateral_conv)
        #     weight_init.c2_xavier_fill(output_conv)
        #     self.add_module("adapter_{}".format(idx + 1), lateral_conv)
        #     self.add_module("layer_{}".format(idx + 1), output_conv)

        #     lateral_convs.append(lateral_conv)
        #     output_convs.append(output_conv)
        # # Place convs into top-down order (from low to high resolution)
        # # to make the top-down computation in forward clearer.
        # self.lateral_convs = lateral_convs[::-1]
        # self.output_convs = output_convs[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        # ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        return ret

    @autocast(enabled=False)
    def forward_features(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        
        # print('????????????????????')
        # for o in out:
        #     print(o.shape)
        
        # feats = []
        # for f in self.in_features[::-1]:
        #     feats.append(features[f])
        
        feats = []
        for f in self.in_features[::-1]:
            feats.append(features[f])
        c4, c3, c2, c1 = feats
        c4, c3, c2 = out
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=c2.size()[2:],mode='nearest')
        
        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=c2.size()[2:],mode='nearest')
        
        _c2 = self.linear_c2(c2)
       
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1)) #(n, c, 128, 128)
        
        _, _, h, w = _c.shape
        nh, nw = 8, 8
        ph, pw = h // nh, w // nw
        
        
        for d in range(self.depth):
            output = []
            output.append(getattr(self, f'short_path_{d}')(_c))
            output.append(F.interpolate(
                    getattr(self, f'image_pool_{d}')(_c),
                    size=_c.size()[2:],
                    mode='nearest'))
            query = rearrange(_c, 'b c (nh ph) (nw pw) -> (b nh nw) (ph pw) c', nh=nh, nw=nw)
            for j, r in enumerate([2, 4, 8]):
                rh = rw = r
                pad_w = [rw//2-1, rw//2] if rw%2 ==0 else [rw//2]*4
                pad_h = [rh//2-1, rh//2] if rh%2 ==0 else [rh//2]*4
                context = F.pad(_c, pad_w+pad_h)
                context = getattr(self, f'ds_{d}')[j](context)
            
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
                context = rearrange(context, 'b c (ph nh) (pw nw) -> b (ph pw) (nh nw c)', nh=rh, nw=rw)
                _output, _ = getattr(self, f'attn_{d}')[j](query, context, context)
                _output = rearrange(_output, '(b nh nw) (ph pw) c -> b c (nh ph) (nw pw)', nh=nh, nw=nw, ph=ph, pw=pw)
                output.append(_output)  
                # output[-1] += _c
            output = getattr(self, f'cat_{d}')(torch.cat(output, dim=1))
            _c = output
                
        _c1 = self.linear_c1(c1)
        output = F.interpolate(output, size=c1.size()[2:], mode='bilinear')
        output = self.low_level_fuse(torch.cat([output, _c1], dim=1))
        return output, None, out