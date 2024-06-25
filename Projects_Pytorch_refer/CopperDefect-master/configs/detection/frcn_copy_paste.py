_base_ = [
    "../../_base_/models/faster_rcnn_r50_fpn.py",
    "../../_base_/datasets/zjdata_1x1split_copy_paste.py",
    "../../_base_/schedules/schedule_1x.py",
    "../../_base_/default_runtime.py"
]

# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)

evaluation = dict(interval=1, metric='mAP',save_best='mAP')

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[17, 19])
runner = dict(type='EpochBasedRunner', max_epochs=24)
work_dir = "/home/dlsuncheng/ZJDetection/Work_dir/20211103/ZJ-FRCN/"
