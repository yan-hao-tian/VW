# VW-Swin/ConvNeXt

## Installation
```
conda create -n vw python=3.8
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.3 -c pytorch -c nvidia
pip install mmcv==1.7.1
pip install -v -e .
```

## Train
More Utilization: See [MMSegmentation Docs](docs).

Swin Transformer
```
tools/dist_train.sh configs/swin/CONFIG.py NUM_GPUS --work-dir work_dirs/EXP_NAME
```
ConvNeXt
```
tools/dist_train.sh configs/convnext/CONFIG.py NUM_GPUS --work-dir work_dirs/EXP_NAME
```

## Evaluation

Single GPU
```
python tools/test.py path/to/config.py path/to/weights.pth --eval mIoU
```
Multiple GPUs
```
tools/dist_test.sh path/to/config.py path/to/weights.pth NUM_GPUS --eval mIoU
```


## <a name="ModelZoo"></a>Model

### ADE20K
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">crop<br/>size</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">mIoU<br/>(ms+flip)</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer_swin_tiny_bs16_160k -->
<!-- ROW: maskformer_swin_base_IN21k_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="configs/swin/vw_swin_base_patch4_window12_640x640_160k_ade20k_pretrain_384x384_22K.py">VW</a></td>
<td align="center">Swin-B</td>
<td align="center">640x640</td>
<td align="center">160k</td>
<td align="center">52.5</td>
<td align="center">53.5</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_swin-b_ade20k/tree/main">model</a></td>
</tr>
<!-- ROW: maskformer_swin_large_IN21k_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="configs/swin/vw_swin_large_patch4_window12_640x640_160k_ade20k_pretrain_384x384_22K.py">VW</a></td>
<td align="center">Swin-L</td>
<td align="center">640x640</td>
<td align="center">160k</td>
<td align="center">54.4</td>
<td align="center">55.8</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_swin-l_ade20k/tree/main">model</a></td>
</tr>
<tr><td align="left"><a href="configs/convnext/vw_convnext_tiny_fp16_512x512_160k_ade20k.py">VW</a></td>
<td align="center">ConvNeXt-T</td>
<td align="center">512x512</td>
<td align="center">160k</td>
<td align="center">47.3</td>
<td align="center">48.3</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_convnext-ti_ade20k/tree/main">model</a></td>
</tr>
<!-- ROW: maskformer_swin_small_bs16_160k -->
 <tr><td align="left"><a href="configs/convnext/vw_convnext_small_fp16_512x512_160k_ade20k.py">VW</a></td>
<td align="center">ConvNeXt-S</td>
<td align="center">512x512</td>
<td align="center">160k</td>
<td align="center">48.8</td>
<td align="center">49.9</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_convnext-s_ade20k/tree/main">model</a></td>
</tr>
<tr><td align="left"><a href="configs/convnext/vw_convnext_base_fp16_640x640_160k_ade20k.py">VW</a></td>
<td align="center">ConvNeXt-B</td>
<td align="center">640x640</td>
<td align="center">160k</td>
<td align="center">53.3</td>
<td align="center">54.1</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_convnext-b_ade20k/tree/main">model</a></td>
</tr>
<!-- ROW: maskformer_swin_small_bs16_160k -->
 <tr><td align="left"><a href="configs/convnext/vw_convnext_large_fp16_640x640_160k_ade20k.py">VW</a></td>
<td align="center">ConvNeXt-L</td>
<td align="center">640x640</td>
<td align="center">160k</td>
<td align="center">54.3</td>
<td align="center">55.1</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_convnext-l_ade20k/tree/main">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/convnext/vw_convnext_xlarge_fp16_640x640_160k_ade20k.py">VW</a></td>
<td align="center">ConvNeXt-XL</td>
<td align="center">640x640</td>
<td align="center">160k</td>
<td align="center">54.6</td>
<td align="center">55.3</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_convnext-xl_ade20k/tree/main">model</a></td>
</tr>
</tbody></table>

### Cityscapes
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">crop<br/>size</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">mIoU<br/>(ms+flip)</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer_swin_tiny_bs16_160k -->
<!-- ROW: maskformer_swin_base_IN21k_384_bs16_160k_res640 -->

<tr><td align="left"><a href="configs/convnext/vw_convnext_tiny_fp16_512x1024_160k_cityscapes.py">VW</a></td>
<td align="center">ConvNeXt-T</td>
<td align="center">512x1024</td>
<td align="center">160k</td>
<td align="center">81.3</td>
<td align="center">48.3</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_convnext-ti_cityscapes/tree/main">model</a></td>
</tr>
<!-- ROW: maskformer_swin_small_bs16_160k -->
 <tr><td align="left"><a href="configs/convnext/vw_convnext_small_fp16_512x1024_160k_cityscapes.py">VW</a></td>
<td align="center">ConvNeXt-S</td>
<td align="center">512x1024</td>
<td align="center">160k</td>
<td align="center">82.2</td>
<td align="center">49.9</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_convnext-s_cityscapes/tree/main">model</a></td>
</tr>
<tr><td align="left"><a href="configs/convnext/vw_convnext_base_fp16_512x1024_160k_cityscapes.py">VW</a></td>
<td align="center">ConvNeXt-B</td>
<td align="center">512x1024</td>
<td align="center">160k</td>
<td align="center">83.2</td>
<td align="center">83.9</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_convnext-b_cityscapes/tree/main">model</a></td>
</tr>
<!-- ROW: maskformer_swin_small_bs16_160k -->
 <tr><td align="left"><a href="configs/convnext/vw_convnext_large_fp16_512x1024_160k_cityscapes.py">VW</a></td>
<td align="center">ConvNeXt-L</td>
<td align="center">512x1024</td>
<td align="center">160k</td>
<td align="center">83.4</td>
<td align="center">84.1</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_convnext-l_cityscapes/tree/main">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/convnext/vw_convnext_xlarge_fp16_512x1024_160k_cityscapes.py">VW</a></td>
<td align="center">ConvNeXt-XL</td>
<td align="center">512x1024</td>
<td align="center">160k</td>
<td align="center">83.6</td>
<td align="center">84.3</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_convnext-xl_cityscapes/tree/main">model</a></td>
</tr>

</tbody></table>

## <a name="CitingVW"></a>Citing VW-Swin/ConvNeXt
```BibTeX
@inproceedings{yan2023multi,
  title={Multi-Scale Representations by Varing Window Attention for Semantic Segmentation},
  author={Yan, Haotian and Wu, Ming and Zhang, Chuang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```