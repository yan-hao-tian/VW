# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import NonLocal2d
import torch.nn.functional as F
from ..builder import HEADS
from .fcn_head import FCNHead

from einops import rearrange

@HEADS.register_module()
class NLHead(FCNHead):
    """Non-local Neural Networks.

    This head is the implementation of `NLNet
    <https://arxiv.org/abs/1711.07971>`_.

    Args:
        reduction (int): Reduction factor of projection transform. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            sqrt(1/inter_channels). Default: True.
        mode (str): The nonlocal mode. Options are 'embedded_gaussian',
            'dot_product'. Default: 'embedded_gaussian.'.
    """

    def __init__(self,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 **kwargs):
        super(NLHead, self).__init__(num_convs=2, **kwargs)
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.nl_block = NonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        
        _, _, h, w = output.shape
        ph, pw = 64, 64
        pad_r = (ph - h % ph) % ph
        pad_b = (pw - w % pw) % pw
        output = F.pad(output, (0, pad_b, 0, pad_r))
        _, _, h_pad, w_pad = output.shape
        nh, nw = h_pad // ph, w_pad // pw
        output = rearrange(output, 'b c (nh ph) (nw pw) -> (b nh nw) c ph pw', nh=nh, nw=nw)
        
        output = self.nl_block(output)
        
        output = rearrange(output, '(b nh nw) c ph pw -> b c (nh ph) (nw pw)', nh=nh, nw=nw)
        output = output[:, :, :h, :w].contiguous()
        
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
