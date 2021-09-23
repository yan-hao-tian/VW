import torch
from mmcv.cnn import NonLocal2d
import torch.nn as nn
import torch.nn.functional as F
from mmseg.ops import resize

from mmcv.cnn import ConvModule, Scale
from ..builder import HEADS
from .fcn_head import FCNHead
from mmcv.cnn import ConvModule
from einops import rearrange
import numpy as np
from numpy.random import rand
import math




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
        self.patch_size = [32, 16, 8, 4]

        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=1,
                padding=1 // 2,
                dilation=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.convs = nn.Sequential(*convs)


        # self.nl_block_1_context = NonLocal2d(
        #     in_channels=self.channels,
        #     reduction=self.reduction,
        #     use_scale=self.use_scale,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     mode=self.mode)
        #     # patch_size=self.patch_size[0])
        # self.nl_block_2_context = NonLocal2d(
        #     in_channels=self.channels,
        #     reduction=self.reduction,
        #     use_scale=self.use_scale,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     mode=self.mode)
        #     # patch_size=self.patch_size[1])
        # self.nl_block_3_context = NonLocal2d(
        #     in_channels=self.channels,
        #     reduction=self.reduction,
        #     use_scale=self.use_scale,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     mode=self.mode)

        self.nl_block_1 = NonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode,
            head=16,
            patch_size=24)
        self.nl_block_2 = NonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode,
            head=16,
            patch_size=12)
        self.nl_block_3 = NonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode,
            head=16,
            patch_size=6)
        
#         self.cam = CAM()
        # self.nl_block_4 = NonLocal2d(
        #     in_channels=self.channels,
        #     reduction=self.reduction,
        #     use_scale=self.use_scale,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=self.norm_cfg,
        #     mode=self.mode)
            # patch_size=self.patch_size[2])

        # self.norm_1 = nn.LayerNorm(self.channels)
        # self.norm_2 = nn.LayerNorm(self.channels)
        # self.norm_3 = nn.LayerNorm(self.channels)
        # self.attn_1 = Attention(in_dim=self.channels, dim=self.channels//2, num_heads=4, qkv_bias=True, attn_drop=0, proj_drop=0, window_size=[self.patch_size[0], self.patch_size[0]])
        # self.attn_2 = Attention(in_dim=self.channels, dim=self.channels//2, num_heads=4, qkv_bias=True, attn_drop=0, proj_drop=0, window_size=[self.patch_size[1], self.patch_size[1]])
        # self.attn_3 = Attention(in_dim=self.channels, dim=self.channels//2, num_heads=4, qkv_bias=True, attn_drop=0, proj_drop=0, window_size=[self.patch_size[2], self.patch_size[2]])
        self.drop_path = DropPath(0)

        # self.norm_0 = nn.LayerNorm(self.channels)
        # self.norm_1 = nn.Identity()
        # self.norm_2 = nn.Identity()
        # self.norm_3 = nn.Identity()

        # self.mlp_1 = Mlp(in_features=self.channels, hidden_features=self.in_channels, out_features=self.channels)
        # self.norm_1 = nn.LayerNorm(self.channels)
        # self.mlp_2 = Mlp(in_features=self.channels, hidden_features=self.in_channels, out_features=self.channels)
        # self.norm_2 = nn.LayerNorm(self.channels)
        # self.mlp_3 = Mlp(in_features=self.channels, hidden_features=self.in_channels, out_features=self.channels)
        # self.norm_3 = nn.LayerNorm(self.channels)

        # self.patch_embedding = ConvModule(
        #         self.in_channels,
        #         self.channels,
        #         kernel_size=2,
        #         stride=2,
        #         padding=2 // 2,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=None,
        #         act_cfg=None)


        self.conv_cat = ConvModule(
                self.channels*5,
                self.channels,
                kernel_size=1,
                padding=1 // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        )
        
        self.pool_conv = ConvModule(
                            self.in_channels, 
                            self.channels, 
                            kernel_size=1, 
                            stride=1,
                            padding=1//2, 
                            conv_cfg=self.conv_cfg, 
                            norm_cfg=self.norm_cfg, 
                            act_cfg=self.act_cfg)


        self.identity = nn.Identity()

        # self.mlp = nn.Sequential(
        #     ConvModule(
        #         self.channels*3,
        #         self.channels,
        #         kernel_size=1,
        #         padding=1 // 2,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg),
        #     ConvModule(
        #         self.channels,
        #         self.channels*3,
        #         kernel_size=1,
        #         padding=1 // 2,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg),
        # )
        
        


    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x) # 2048->512
        n, c, h, w = output.shape
        
        ###NOTE: GAP branch###
        output_p = self.identity(x)
        output_p = resize(
                self.image_pool(output_p),
                size=(h, w),
                mode='bilinear',
                align_corners=self.align_corners)
        
        # h, w = 72, 144 
        # output = resize(
        #     input=output,
        #     size=[h, w],
        #     mode='bilinear',
        #     align_corners=self.align_corners
        # )
        # output_attn = self.identity(output) 
        query_attn = self.pool_conv(x)
        context_attn = F.avg_pool2d(query_attn, kernel_size=4, stride=4)
        
#         output_c = self.cam(query_attn)
#         context_attn_ds4 = F.avg_pool2d(query_attn, kernel_size=4, stride=4)
#         context_attn_ds2 = F.avg_pool2d(query_attn, kernel_size=2, stride=2)
        
        # avgpool the channel
#         query_attn_transpose = query_attn.reshape(*query_attn.size()[:2], -1).transpose(1,2) 
# #         print(query_attn_transpose.shape)
        
#         context_attn_ds4 = F.avg_pool1d(query_attn_transpose, kernel_size=16, stride=16).transpose(1,2).reshape(query_attn.size()[0], query_attn.size()[1]//16, *query_attn.size()[2:]) # (n, 32, h, w)
#         context_attn_ds4 = F.unfold(context_attn_ds4, kernel_size=4, stride=4)
#         context_attn_ds4 = rearrange(context_attn_ds4, 'b (c psh psw) (pnh pnw) -> b c pnh pnw psh psw', psh=4, psw=4, pnh=query_attn.size()[2]//4, pnw=query_attn.size()[3]//4)
#         context_attn_ds4 = rearrange(context_attn_ds4, 'b c pnh pnw psh psw -> b (psh psw c) pnh pnw', psh=4, psw=4, pnh=query_attn.size()[2]//4, pnw=query_attn.size()[3]//4) # (n, 512, h//4, w//4)
 
#         context_attn_ds2 = F.avg_pool1d(query_attn_transpose, kernel_size=4, stride=4).transpose(1,2).reshape(query_attn.size()[0], query_attn.size()[1]//4, *query_attn.size()[2:]) # (n, 128, h, w)
#         context_attn_ds2 = F.unfold(context_attn_ds2, kernel_size=2, stride=2)
#         context_attn_ds2 = rearrange(context_attn_ds2, 'b (c psh psw) (pnh pnw) -> b c pnh pnw psh psw', psh=2, psw=2, pnh=query_attn.size()[2]//2, pnw=query_attn.size()[3]//2)
#         context_attn_ds2 = rearrange(context_attn_ds2, 'b c pnh pnw psh psw -> b (psh psw c) pnh pnw', psh=2, psw=2, pnh=query_attn.size()[2]//2, pnw=query_attn.size()[3]//2) # (n, 512, h//4, w//4)
        
#         context_attn_ds1 = query_attn
        # x = self.identity(output_attn)
        # x = self.pool_conv(x)
        #output_gap = self.image_pool(x)
        #output_gap = resize(
        #EXPLICIT SPARSE TRANSFORMER    input=output_gap,
        #    size=[h, w],
        #    mode='bilinear',
        #    align_corners=self.align_corners
        #)

#         patch_size=16
#         patch_num_h, patch_num_w = h//patch_size, w//patch_size
        
#         query = F.unfold(query_attn, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
#         query = rearrange(query, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=patch_size, psw=patch_size, pnh=patch_num_h, pnw=patch_num_w)
        
#         context_ds4 = F.unfold(context_attn_ds4, kernel_size=(patch_size, patch_size), stride=4, padding=6) 
#         context_ds4 = rearrange(context_ds4, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=patch_size, psw=patch_size, pnh=patch_num_h, pnw=patch_num_w)
        
#         context_ds2 = F.unfold(context_attn_ds2, kernel_size=(patch_size, patch_size), stride=8, padding=4) 
#         context_ds2 = rearrange(context_ds2, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=patch_size, psw=patch_size, pnh=patch_num_h, pnw=patch_num_w)
        
#         context_ds1 = F.unfold(context_attn_ds1, kernel_size=(patch_size, patch_size), stride=16, padding=0) 
#         context_ds1 = rearrange(context_ds1, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=patch_size, psw=patch_size, pnh=patch_num_h, pnw=patch_num_w)
        
#         output_1 = self.nl_block_1(context_ds4, query)
#         output_1 = rearrange(output_1, '(b pnh pnw) c psh psw -> b c (pnh psh) (pnw psw)', pnh=patch_num_h, pnw=patch_num_w, psh=patch_size, psw=patch_size)
        
#         output_2 = self.nl_block_2(context_ds2, query)
#         output_2 = rearrange(output_2, '(b pnh pnw) c psh psw -> b c (pnh psh) (pnw psw)', pnh=patch_num_h, pnw=patch_num_w, psh=patch_size, psw=patch_size)
        
#         output_3 = self.nl_block_3(context_ds1, query)
#         output_3 = rearrange(output_3, '(b pnh pnw) c psh psw -> b c (pnh psh) (pnw psw)', pnh=patch_num_h, pnw=patch_num_w, psh=patch_size, psw=patch_size)

        ###NOTE:The branch 1###
        
        patch_size = self.patch_size[0]
        patch_num_h, patch_num_w = h // patch_size, w // patch_size
        context_1 = F.unfold(context_attn, kernel_size=(24, 24), stride=8, padding=8) # x.shape is (b, c*48*48, 2, 4)
        context_1 = rearrange(context_1, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=24, psw=24, pnh=patch_num_h, pnw=patch_num_w)
        query_1 = F.unfold(query_attn, kernel_size=(patch_size, patch_size), stride=patch_size)
        query_1 = rearrange(query_1, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=patch_size, psw=patch_size, pnh=patch_num_h, pnw=patch_num_w)

#         output_1_context = F.adaptive_avg_pool2d(output_1, (3,3)) # (48, 48) -> (bs, c, 3, 3)
#         output_1_context = self.nl_block_1_context(output_attn_1, output_1_context) # -> (bs, c, 3, 3) 
#         # NOTE
#         # output_1_context = self.nl_block_1_context(output_1_context, output_attn_1) # -> (bs, c, 32, 32) 
#         # output_1_context = rearrange(output_1_context, '(b pnh pnw) c psh psw -> b c (pnh psh) (pnw psw)', pnh=patch_num_h, pnw=patch_num_w, psh=patch_size, psw=patch_size) 
#         output_1_context_center = output_1_context[:,:,1,1].unsqueeze(-1).unsqueeze(-1)
#         output_1_context = F.unfold(output_1_context, kernel_size=(1,1), stride=(1,1)) 
#         output_1_context = rearrange(output_1_context, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=1, psw=1, pnh=3, pnw=3) # -> (bs*9, c, 1, 1)
#         output_1 = F.unfold(output_1, kernel_size=(16,16), stride=(16,16)) # -> (bs*9, c, 16, 16)
#         output_1 = rearrange(output_1, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=16, psw=16, pnh=3, pnw=3)
#         output_1 = output_1+output_1_context
#         # output_1 = output_1+output_1_context
#         output_1 = rearrange(output_1, '(b pnh pnw) c psh psw -> b c (pnh psh) (pnw psw)', pnh=3, pnw=3, psh=16, psw=16)
#         output_1[:,:,16:32,16:32]-=output_1_context_center
         

        output_1 = self.nl_block_1(context_1, query_1) # 512 -> 256 -> 512
        output_1 = rearrange(output_1, '(b pnh pnw) c psh psw -> b c (pnh psh) (pnw psw)', pnh=patch_num_h, pnw=patch_num_w, psh=patch_size, psw=patch_size) 
        output_1 = resize(output_1, size=(h, w), mode='bilinear', align_corners=self.align_corners)

        ###NOTE:The branch 2###

        patch_size = self.patch_size[1]
        patch_num_h, patch_num_w = h // patch_size, w // patch_size
        context_2 = F.unfold(context_attn, kernel_size=(12, 12), stride=4, padding=4)
        context_2 = rearrange(context_2, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=12, psw=12, pnh=patch_num_h, pnw=patch_num_w)
        query_2 = F.unfold(query_attn, kernel_size=(patch_size, patch_size), stride=patch_size)
        query_2 = rearrange(query_2, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw ', psh=patch_size, psw=patch_size, pnh=patch_num_h, pnw=patch_num_w)

#         output_2_context = F.adaptive_avg_pool2d(output_2, (3,3)) # (48, 48) -> (bs, c, 3, 3)
#         output_2_context = self.nl_block_2_context(output_attn_2, output_2_context)
#         output_2_context_center = output_2_context[:,:,1,1].unsqueeze(-1).unsqueeze(-1)
#         output_2_context = F.unfold(output_2_context, kernel_size=(1,1), stride=(1,1))
#         output_2_context = rearrange(output_2_context, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=1, psw=1, pnh=3, pnw=3)
#         output_2 = F.unfold(output_2, kernel_size=(8,8), stride=(8,8))
#         output_2 = rearrange(output_2, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=8, psw=8, pnh=3, pnw=3)
#         output_2 = output_2+output_2_context
#         # output_2 = output_2+torch.sigmoid(output_2_context)
#         output_2 = rearrange(output_2, '(b pnh pnw) c psh psw -> b c (pnh psh) (pnw psw)', pnh=3, pnw=3, psh=8, psw=8)
#         output_2[:,:,8:16,8:16]-=output_2_context_center

        output_2 = self.nl_block_2(context_2, query_2) # 512 -> 256 -> 512
        output_2 = rearrange(output_2, '(b pnh pnw) c psh psw -> b c (pnh psh) (pnw psw)', pnh=patch_num_h, pnw=patch_num_w, psh=patch_size, psw=patch_size) 
        output_2 = resize(output_2, size=(h, w), mode='bilinear', align_corners=self.align_corners)

        ###NOTE:The branch 3###

        patch_size = self.patch_size[2]
        patch_num_h, patch_num_w = h // patch_size, w // patch_size
        context_3 = F.unfold(context_attn, kernel_size=(6, 6), stride=2, padding=2) 
        context_3 = rearrange(context_3, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw ', psh=6, psw=6, pnh=patch_num_h, pnw=patch_num_w)
        query_3 = F.unfold(query_attn, kernel_size=(patch_size, patch_size), stride=patch_size)
        query_3 = rearrange(query_3, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw ', psh=patch_size, psw=patch_size, pnh=patch_num_h, pnw=patch_num_w)

#         output_3_context = F.adaptive_avg_pool2d(output_3, (3,3)) # (48, 48) -> (bs, c, 3, 3)
#         output_3_context = self.nl_block_3_context(output_attn_3, output_3_context)
#         output_3_context_center = output_3_context[:,:,1,1].unsqueeze(-1).unsqueeze(-1)
#         output_3_context = F.unfold(output_3_context, kernel_size=(1,1), stride=(1,1))
#         output_3_context = rearrange(output_3_context, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=1, psw=1, pnh=3, pnw=3)
#         output_3 = F.unfold(output_3, kernel_size=(4,4), stride=(4,4))
#         output_3 = rearrange(output_3, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw', psh=4, psw=4, pnh=3, pnw=3)
#         output_3 = output_3+output_3_context
#         # output_3 = output_3+torch.sigmoid(output_3_context)
#         output_3 = rearrange(output_3, '(b pnh pnw) c psh psw -> b c (pnh psh) (pnw psw)', pnh=3, pnw=3, psh=4, psw=4)
#         output_3[:,:,4:8,4:8]-=output_3_context_center

        output_3 = self.nl_block_3(context_3, query_3) # 512 -> 256 -> 512
        output_3 = rearrange(output_3, '(b pnh pnw) c psh psw -> b c (pnh psh) (pnw psw)', pnh=patch_num_h, pnw=patch_num_w, psh=patch_size, psw=patch_size) 
        output_3 = resize(output_3, size=(h, w), mode='bilinear', align_corners=self.align_corners)

        
        # output_a = self.mlp(torch.cat([output_1, output_2, output_3], dim=1))+torch.cat([output_1, output_2, output_3], dim=1)

        ###NOTE:The branch 4###

        # patch_size = self.patch_size[3]
        # patch_num_h, patch_num_w = h // patch_size, w // patch_size
        # # output_3 = F.pad(x, pad=(2,2,2,2), mode='constant', value=0)
        # output_4 = F.pad(x, pad=(2,2,2,2), mode='constant', value=0)
        # # output_3 = F.adaptive_avg_pool2d(output_attn, (h//2+8, w//2+8))
        # # output_3 = F.unfold(output_3, kernel_size=(patch_size, patch_size), stride=patch_size//2) # x.shape is (b, c*32*32, 2, 4)
        # output_4 = F.unfold(output_4, kernel_size=(patch_size+patch_size//2, patch_size+patch_size//2), stride=patch_size//2)
        # output_attn_4 = F.unfold(output_attn, kernel_size=(patch_size, patch_size), stride=patch_size)
        # output_attn_4 = rearrange(output_attn_4, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw ', psh=patch_size, psw=patch_size, pnh=patch_num_h, pnw=patch_num_w)
        # # output_3 = rearrange(output_3, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw ', psh=patch_size, psw=patch_size, pnh=patch_num_h, pnw=patch_num_w)
        # output_4 = rearrange(output_4, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw ', psh=patch_size+patch_size//2, psw=patch_size+patch_size//2, pnh=patch_num_h, pnw=patch_num_w)
        # # output_3 = output_3+self.drop_path(self.attn_3(self.norm_3(output_3)))
        # # output_ = output_+self.positionalencoding2d(512, patch_size, patch_size).repeat(1, 1, 1).cuda()
        # output_4 = self.nl_block_4(output_4, output_attn_4) # 512 -> 256 -> 512
        # output_4 = rearrange(output_4, '(b pnh pnw) c psh psw -> b c (pnh psh) (pnw psw)', pnh=patch_num_h, pnw=patch_num_w, psh=patch_size, psw=patch_size) 
        # output_4 = resize(output_4, size=(h, w), mode='bilinear', align_corners=self.align_corners)

        # output_a = self.mlp(torch.cat([output_1, output_2, output_3], dim=1))+torch.cat([output_1, output_2, output_3], dim=1)

        if self.concat_input:
            output = self.conv_cat(torch.cat([output, output_1, output_2, output_3, output_p], dim=1))
        output = self.cls_seg(output)
        return output


    def positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe

    # def adaptive_context_vector(self, x, patch_size):
    #     # input_x: n, c, h, w
    #     bs, c, h, w = x.shape
    #     pad = patch_size 
    #     kernel_size = patch_size * 3
    #     stride = patch_size
    #     patch_num_h = h//patch_size
    #     patch_num_w = w//patch_size
    #     x_c_size = patch_size*3//4
    #     x_d_size = patch_size*2//4
    #     x_e_size = patch_size*1//4

    #     bs = patch_num_h*patch_num_w*bs

    #     x = F.unfold(x, kernel_size=kernel_size, padding=pad, stride=stride)
    #     x = rearrange(x, 'b (c psh psw) (pnh pnw) -> (b pnh pnw) c psh psw ', psh=patch_size, psw=patch_size, pnh=patch_num_h, pnw=patch_num_w)
    #     x_c = F.adaptive_avg_pool2d(x[:, :, patch_size:2*patch_size, patch_size:2*patch_size], output_size=x_c_size).view(bs, c, -1) # (bs, c, n)
    #     x_d = torch.cat([
    #         F.adaptive_avg_pool2d(x[:, :, patch_size//2:patch_size, patch_size//2:-patch_size//2], output_size=x_d_size).view(bs, c, -1),
    #         F.adaptive_avg_pool2d(x[:, :, -patch_size:-patch_size//2, patch_size//2:-patch_size//2], output_size=x_d_size).view(bs, c, -1),
    #         F.adaptive_avg_pool2d(x[:, :, patch_size:-patch_size, patch_size//2:patch_size], output_size=x_d_size).view(bs, c, -1),
    #         F.adaptive_avg_pool2d(x[:, :, patch_size:-patch_size, -patch_size:-patch_size//2], output_size=x_d_size).view(bs, c, -1)
    #     ], dim=-1)
    #     x_e = torch.cat([
    #         F.adaptive_avg_pool2d(x[:, :, :patch_size//2,:], output_size=x_e_size).view(bs, c, -1),
    #         F.adaptive_avg_pool2d(x[:, :, -patch_size//2:, :], output_size=x_e_size).view(bs, c, -1),
    #         F.adaptive_avg_pool2d(x[:, :, patch_size//2:-patch_size//2, :patch_size//2], output_size=x_e_size).view(bs, c, -1),
    #         F.adaptive_avg_pool2d(x[:, :, patch_size//2:-patch_size//2, -patch_size//2:], output_size=x_e_size).view(bs, c, -1)
    #     ], dim=-1)

    #     return torch.cat([x_c, x_d, x_e], dim=-1)

        


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

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

class Attention(nn.Module):
    def __init__(self, in_dim, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=[32,32]):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(in_dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.window_size = window_size

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads), requires_grad=True)  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B N N)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.head_dim*self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class CAM(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self):
        super(CAM, self).__init__()
        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)

        out = self.gamma(out) + x
        return out
