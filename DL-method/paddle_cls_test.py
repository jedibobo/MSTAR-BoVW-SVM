import paddlex as pdx
from paddlex import transforms as T

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
train_transforms = T.Compose(
    [T.RandomCrop(crop_size=158), T.RandomHorizontalFlip(), T.Normalize()])

eval_transforms = T.Compose([
    T.ResizeByShort(short_size=158), T.CenterCrop(crop_size=158), T.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
train_dataset = pdx.datasets.ImageNet(
    data_dir='ALL',
    file_list='ALL/train_list.txt',
    label_list='ALL/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.ImageNet(
    data_dir='ALL',
    file_list='ALL/val_list.txt',
    label_list='ALL/labels.txt',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/visualdl.md
num_classes = len(train_dataset.labels)
model = pdx.cls.PPLCNet(num_classes=num_classes, scale=1)

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/models/classification.md
# 各参数介绍与调整说明：https://github.com/PaddlePaddle/PaddleX/tree/develop/docs/parameters.md
model.train(
    num_epochs=50,
    pretrain_weights='IMAGENET',
    train_dataset=train_dataset,
    train_batch_size=64,
    eval_dataset=eval_dataset,
    lr_decay_epochs=[4, 6, 8],
    learning_rate=0.1,
    save_dir='output/pplcnet',
    log_interval_steps=100,
    label_smoothing=.1,
    save_interval_epochs=5,
    use_vdl=True)