# Lawin Transformer

[Paper](https://arxiv.org/abs/2201.01615)

Lawin Transformer: Improving Semantic Segmentation Transformer with Multi-Scale Representations via Large Window Attention.<br>


## Installation

For install and data preparation, please refer to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Important requirements: ```CUDA 11.6``` and  ```pytorch 1.8.1``` 

```
pip install torchvision==0.9.1
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
pip install einops
cd lawin && pip install -e . --user
```

## Evaluation

```
# Single-gpu testing
python tools/test.py local_configs/segformer/B2/lawin.b2.512x512.ade.160k.py /path/to/checkpoint_file
```

## Training

Download [weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) pretrained on ImageNet-1K, and put them in a folder ```pretrained/```.

Example: train ```SegFormer-B2``` on ```ADE20K```:

```
# Multi-gpu training
./tools/dist_train.sh local_configs/segformer/B2/segformer.b2.512x512.ade.160k.py <GPU_NUM> --work-dir <WORK_DIRS_TO_SAVE_WEIGHTS&LOGS> 
```

## Citation
```
@article{yan2022lawin,
  title={Lawin transformer: Improving semantic segmentation transformer with multi-scale representations via large window attention},
  author={Yan, Haotian and Zhang, Chuang and Wu, Ming},
  journal={arXiv preprint arXiv:2201.01615},
  year={2022}
}
```
