_base_ = [
    '../_base_/datasets/mastr1325.py',
    '../_base_/default_runtime.py',
]

# The class_weight is borrowed from https://github.com/openseg-group/OCNet.pytorch/issues/14 # noqa
# Licensed under the MIT License
# class_weight = [
#     0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
#     1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
#     1.0507
# ]
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/ddrnet/pretrain/ddrnet23-in1kpre_3rdparty-9ca29f62.pth'  # noqa
checkpoint = 'E:\\daity\\mmsegmentation-at-af\\mmsegmentation\\checkpoints\\DDRNet-CA3-7993-AdamW--iter_285000.pth'
# load_from = 'E:\\daity\\mmsegmentation-at-af\\mmsegmentation\\checkpoints\\DDRNet-9605-lars-300k-iter_231000.pth'
# resume = True
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='LTbANet',
        in_channels=3,
        channels=64,
        ppm_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        type='DDRHead',
        in_channels=64 * 4,
        channels=128,
        dropout_ratio=0.,
        num_classes=5,
        align_corners=False,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                # class_weight=class_weight,
                loss_weight=1.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                # class_weight=class_weight,
                loss_weight=0.4),
        ]),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_dataloader = dict(batch_size=4, num_workers=4)

iters = 300000
# optimizer
optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]

# training schedule for 120k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=iters // 100),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=304)
