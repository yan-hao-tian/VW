_base_ = [
    '../_base_/models/vw_convnext.py', 
    '../_base_/datasets/ade20k_640x640.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (640, 640)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth'  # noqa
model = dict(
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='large',
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=150,
        channels=512,
    ),
    # auxiliary_head=dict(in_channels=768, num_classes=150),
    test_cfg = dict(mode='whole'))

# stage decay learning rate
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    })

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 4 GPUs with 4 images per GPU
data = dict(samples_per_gpu=4)
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()

runner = dict(type='IterBasedRunner', max_iters=160000)
# saving checkpoints every 8000 iters
checkpoint_config = dict(by_epoch=False, interval=8000)
# evaluating every 1000 iters and save the best.
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
