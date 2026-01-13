#训练的时候报了找不到验证集的问题，现在我试试能否正常加载验证集
# python train.py \
#   --train-img-dir /root/autodl-tmp/train2014\
#   --train-ann-file /root/autodl-tmp/annotations/instances_train2014.json \
#   --val-img-dir /root/autodl-tmp/val2014\ \
#   --val-ann-file /root/autodl-tmp/annotations/instances_val2014.json \
#   --batch-size 4 \
#   --epochs 1 \
#   --lr 1e-5 \
#   --device cuda

import os
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# 导入自定义的数据加载器
from dataloader import COCOSegDataset, SemanticSegTransform

# 设置路径
VAL_IMG_DIR = '/root/autodl-tmp/val2014'
VAL_ANN_FILE = '/root/autodl-tmp/annotations/instances_val2014.json'

# 检查路径是否存在
print(f"检查路径是否存在:")
print(f"验证图像目录: {VAL_IMG_DIR} - {'存在' if os.path.exists(VAL_IMG_DIR) else '不存在'}")
print(f"验证注释文件: {VAL_ANN_FILE} - {'存在' if os.path.exists(VAL_ANN_FILE) else '不存在'}")

# 创建变换
val_transform = SemanticSegTransform(
    resize_size=(256, 256),
    flip_prob=0.0  # 验证时不使用翻转
)

# 创建验证数据集
print("\n尝试创建验证数据集...")
try:
    val_dataset = COCOSegDataset(
        img_dir=VAL_IMG_DIR,
        ann_file=VAL_ANN_FILE,
        transform=val_transform
    )
    print(f"验证数据集创建成功，大小: {len(val_dataset)}")
    
    # 尝试加载几个样本
    print("\n尝试加载前3个样本...")
    for i in range(3):
        try:
            image, mask = val_dataset[i]
            print(f"样本 {i+1} 加载成功:")
            print(f"  图像形状: {image.shape}")
            print(f"  掩码形状: {mask.shape}")
        except Exception as e:
            print(f"样本 {i+1} 加载失败: {e}")
    
    print("\n验证集加载测试完成！")
    
except Exception as e:
    print(f"验证数据集创建失败: {e}")
    import traceback
    traceback.print_exc()