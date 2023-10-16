# Lawin Transformer: Improving New-Era Vision Backbones with Multi-Scale Representations for Semantic Segmentation

[[`Paper`](https://drive.google.com/file/d/1eQd_bj1DTNTxpMVvfIzkt0vdZujrf9IK/view)] [[`Poster`](https://drive.google.com/file/d/1ifjjedbX8nbF0iBDI7g3uL_j8Hkf84sN/view)] [[`Supplementary`](https://drive.google.com/file/d/1QPFqAS-3LETbRULfPk-vfoO50g_MfPAN/view)] [[`Code`](https://github.com/yan-hao-tian/lawin/blob/70903a10403d4d8b87b0a2fe39a7cf045cf5a476/mask_former/modeling/heads/pixel_decoder.py#L196)]

🔥🔥🔥 Accepted by [Transformers for Vision @CVPR2023](https://sites.google.com/view/t4v-cvpr23/home).

## Installation

See [installation instructions](INSTALL.md).

## Datasets

See [Preparing Datasets for MaskFormer](datasets/README.md).

## Train

More Utilization: See [Getting Started with MaskFormer](GETTING_STARTED.md). 
### MaskFormer
Swin-Tiny
```
python ./train_net.py \
--resume --num-gpus 2 --dist-url auto \
--config-file configs/ade20k-150/swin/lawin/lawin_maskformer_swin_tiny_bs16_160k.yaml \
OUTPUT_DIR path/to/tiny TEST.EVAL_PERIOD 10000 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```
Swin-Small
```
python ./train_net.py \
--resume --num-gpus 4 --dist-url auto \
--config-file configs/ade20k-150/swin/lawin/lawin_maskformer_swin_small_bs16_160k.yaml \
OUTPUT_DIR path/to/small TEST.EVAL_PERIOD 10000 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```

Swin-Base
```
python ./train_net.py \
--resume --num-gpus 8 --dist-url auto \
--config-file configs/ade20k-150/swin/lawin/lawin_maskformer_swin_base_IN21k_384_bs16_160k_res640.yaml \
OUTPUT_DIR path/to/base TEST.EVAL_PERIOD 10000 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```

Swin-Large
```
python ./train_net.py \
--resume --num-gpus 16 --dist-url auto \
--config-file configs/ade20k-150/swin/lawin/lawin_maskformer_swin_large_IN21k_384_bs16_160k_res640.yaml \
OUTPUT_DIR path/to/large TEST.EVAL_PERIOD 10000 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```
### Mask2Former
Swin-Tiny
```
cd Mask2Former
python ./train_net.py \
--resume --num-gpus 2 --dist-url auto \
--config-file configs/ade20k/semantic-segmentation/swin/lawin/lawin_maskformer2_swin_tiny_bs16_160k.yaml \
OUTPUT_DIR path/to/tiny TEST.EVAL_PERIOD 10000 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```
Swin-Small
```
cd Mask2Former
python ./train_net.py \
--resume --num-gpus 4 --dist-url auto \
--config-file configs/ade20k/semantic-segmentation/swin/lawin/lawin_maskformer2_swin_small_bs16_160k.yaml \
OUTPUT_DIR path/to/small TEST.EVAL_PERIOD 10000 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```

Swin-Base
```
cd Mask2Former
python ./train_net.py \
--resume --num-gpus 8 --dist-url auto \
--config-file configs/ade20k/semantic-segmentation/swin/lawin/lawin_maskformer2_swin_base_IN21k_384_bs16_160k_res640.yaml \
OUTPUT_DIR path/to/base TEST.EVAL_PERIOD 10000 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```

Swin-Large
```
cd Mask2Former
python ./train_net.py \
--resume --num-gpus 16 --dist-url auto \
--config-file configs/ade20k/semantic-segmentation/swin/lawin/lawin_maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml \
OUTPUT_DIR path/to/large TEST.EVAL_PERIOD 10000 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```
## Evaluation
### MaskFormer
```
python ./train_net.py \
--eval-only --num-gpus NGPUS --dist-url auto \
--config-file path/to/config \
MODEL.WEIGHTS path/to/weight TEST.AUG.ENABLED True MODEL.MASK_FORMER.SIZE_DIVISIBILITY 64
```

### Mask2Former
```
cd Mask2Former
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
 <tr><td align="left"><a href="configs/ade20k-150/swin/lawin/lawin_maskformer_swin_tiny_bs16_160k.yaml">Lawin-MaskFormer</a></td>
<td align="center">Swin-T</td>
<td align="center">512x512</td>
<td align="center">160k</td>
<td align="center">47.4</td>
<td align="center">49.0</td>
<td align="center"><a href="https://drive.google.com/file/d/1kzu8K8phAEPo6NHoLubSePlKl6N86Mde/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: maskformer_swin_small_bs16_160k -->
 <tr><td align="left"><a href="configs/ade20k-150/swin/lawin/lawin_maskformer_swin_small_bs16_160k.yaml">Lawin-MaskFormer</a></td>
<td align="center">Swin-S</td>
<td align="center">512x512</td>
<td align="center">160k</td>
<td align="center">50.5</td>
<td align="center">52.7</td>
<td align="center"><a href="https://drive.google.com/file/d/1mPmwVUlJckeldTwpsKIiRaE5wQOpKaou/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: maskformer_swin_base_IN21k_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="configs/ade20k-150/swin/lawin/lawin_maskformer_swin_base_IN21k_384_bs16_160k_res640.yaml">Lawin-MaskFormer</a></td>
<td align="center">Swin-B</td>
<td align="center">640x640</td>
<td align="center">160k</td>
<td align="center">53.8</td>
<td align="center">54.6</td>
<td align="center"><a href="https://drive.google.com/file/d/1Llvp_-KsVdV9pK1yBG29D3xhryrYOeVf/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: maskformer_swin_large_IN21k_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="configs/ade20k-150/swin/lawin/lawin_maskformer_swin_large_IN21k_384_bs16_160k_res640.yaml">Lawin-MaskFormer</a></td>
<td align="center">Swin-L</td>
<td align="center">640x640</td>
<td align="center">160k</td>
<td align="center">55.3</td>
<td align="center">56.5</td>
<td align="center"><a href="https://drive.google.com/file/d/14paNql4Mu1ukRB-k4ewyR8dV00R0MIFo/view?usp=sharing">model</a></td>
</tr>
</tbody></table>

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
<!-- ROW: mask2former_swin_tiny_bs16_160k -->
 <tr><td align="left"><a href="Mask2Former/configs/ade20k/semantic-segmentation/swin/lawin/lawin_maskformer2_swin_tiny_bs16_160k.yaml">Lawin-Mask2Former</a></td>
<td align="center">Swin-T</td>
<td align="center">512x512</td>
<td align="center">160k</td>
<td align="center">48.2</td>
<td align="center">50.5</td>
<td align="center"><a href="https://drive.google.com/file/d/1MCCg-I0NE4boOKTDXQDDiOv720ZVJW1a/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: mask2former_swin_small_bs16_160k -->
 <tr><td align="left"><a href="Mask2Former/configs/ade20k/semantic-segmentation/swin/lawin/lawin_maskformer2_swin_small_bs16_160k.yaml">Lawin-Mask2Former</a></td>
<td align="center">Swin-S</td>
<td align="center">512x512</td>
<td align="center">160k</td>
<td align="center">52.1</td>
<td align="center">53.7</td>
<td align="center"><a href="https://drive.google.com/file/d/139MKGZwoYkiYPQtjNtrau6GjMAtzeApp/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: mask2former_swin_base_IN21k_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="Mask2Former/configs/ade20k/semantic-segmentation/swin/lawin/lawin_maskformer2_swin_base_IN21k_384_bs16_160k_res640.yaml">Lawin-Mask2Former</a></td>
<td align="center">Swin-B</td>
<td align="center">640x640</td>
<td align="center">160k</td>
<td align="center">54.6</td>
<td align="center">56.0</td>
<td align="center"><a href="https://drive.google.com/file/d/1Xa5gW981iLZ3feFBVqS-lvFa9e8Z8ZMP/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: mask2former_swin_large_IN21k_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="Mask2Former/configs/ade20k/semantic-segmentation/swin/lawin/lawin_maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml">Lawin-Mask2Former</a></td>
<td align="center">Swin-L</td>
<td align="center">640x640</td>
<td align="center">160k</td>
<td align="center">56.5</td>
<td align="center">57.8</td>
<td align="center"><a href="https://drive.google.com/file/d/1aMxSCPXWxfidx7wU4LBr7WkF59N4TyY7/view?usp=sharing">model</a></td>
</tr>
</tbody></table>

## <a name="CitingMaskFormer"></a>Citing Lawin Transformer

<!-- If you use Lawin Transformer in your research or wish to refer to the baseline results published in the [Model Zoo]((#ModelZoo)), please use the following BibTeX entry. -->

```BibTeX
@article{yan2022lawin,
  title={Lawin transformer: Improving semantic segmentation transformer with multi-scale representations via large window attention},
  author={Yan, Haotian and Zhang, Chuang and Wu, Ming},
  journal={arXiv preprint arXiv:2201.01615},
  year={2022}
}
```
