# VW-MaskFormer

## Installation

See [installation instructions](INSTALL.md).

## Datasets

See [Preparing Datasets for MaskFormer](datasets/README.md).

## Train

More Utilization: See [Getting Started with MaskFormer](GETTING_STARTED.md). 

Swin-Tiny
```
python ./train_net.py \
--resume --num-gpus 2 --dist-url auto \
--config-file configs/ade20k-150/swin/vw/vw_maskformer_swin_tiny_bs16_160k.yaml \
OUTPUT_DIR path/to/tiny TEST.EVAL_PERIOD 10000 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```
Swin-Small
```
python ./train_net.py \
--resume --num-gpus 4 --dist-url auto \
--config-file configs/ade20k-150/swin/vw/vw_maskformer_swin_small_bs16_160k.yaml \
OUTPUT_DIR path/to/small TEST.EVAL_PERIOD 10000 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```

Swin-Base
```
python ./train_net.py \
--resume --num-gpus 8 --dist-url auto \
--config-file configs/ade20k-150/swin/vw/vw_maskformer_swin_base_IN21k_384_bs16_160k_res640.yaml \
OUTPUT_DIR path/to/base TEST.EVAL_PERIOD 10000 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```

Swin-Large
```
python ./train_net.py \
--resume --num-gpus 16 --dist-url auto \
--config-file configs/ade20k-150/swin/vw/vw_maskformer_swin_large_IN21k_384_bs16_160k_res640.yaml \
OUTPUT_DIR path/to/large TEST.EVAL_PERIOD 10000 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```
## Evaluation
```
cd MaskFormer
python ./train_net.py \
--eval-only --num-gpus NGPUS --dist-url auto \
--config-file path/to/config \
MODEL.WEIGHTS path/to/weight TEST.AUG.ENABLED True MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```


## <a name="ModelZoo"></a>Model
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
 <tr><td align="left"><a href="configs/ade20k-150/swin/vw/vw_maskformer_swin_tiny_bs16_160k.yaml">VW-MaskFormer</a></td>
<td align="center">Swin-T</td>
<td align="center">512x512</td>
<td align="center">160k</td>
<td align="center">47.4</td>
<td align="center">49.0</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_maskformer-ti_ade20k/tree/main">model</a></td>
</tr>
<!-- ROW: maskformer_swin_small_bs16_160k -->
 <tr><td align="left"><a href="configs/ade20k-150/swin/vw/vw_maskformer_swin_small_bs16_160k.yaml">VW-MaskFormer</a></td>
<td align="center">Swin-S</td>
<td align="center">512x512</td>
<td align="center">160k</td>
<td align="center">50.5</td>
<td align="center">52.7</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_maskformer-s_ade20k/tree/main">model</a></td>
</tr>
<!-- ROW: maskformer_swin_base_IN21k_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="configs/ade20k-150/swin/vw/vw_maskformer_swin_base_IN21k_384_bs16_160k_res640.yaml">VW-MaskFormer</a></td>
<td align="center">Swin-B</td>
<td align="center">640x640</td>
<td align="center">160k</td>
<td align="center">53.8</td>
<td align="center">54.6</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_maskformer-b_ade20k/tree/main">model</a></td>
</tr>
<!-- ROW: maskformer_swin_large_IN21k_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="configs/ade20k-150/swin/vw/vw_maskformer_swin_large_IN21k_384_bs16_160k_res640.yaml">VW-MaskFormer</a></td>
<td align="center">Swin-L</td>
<td align="center">640x640</td>
<td align="center">160k</td>
<td align="center">55.3</td>
<td align="center">56.5</td>
<td align="center"><a href="https://huggingface.co/yan-hao-tian/vw_maskformer-l_ade20k/tree/main">model</a></td>
</tr>
</tbody></table>

## <a name="CitingVW"></a>Citing VW-MaskFormer
```BibTeX
@inproceedings{yan2023multi,
  title={Multi-Scale Representations by Varing Window Attention for Semantic Segmentation},
  author={Yan, Haotian and Wu, Ming and Zhang, Chuang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```