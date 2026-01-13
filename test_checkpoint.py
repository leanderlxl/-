# 测试checkpoint加载功能

import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 创建一个简单的DeepLabv3模型
print("创建模型...")
model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=81)

# 创建优化器
params_to_optimize = [param for param in model.parameters() if param.requires_grad]
optimizer = optim.Adam(params_to_optimize, lr=0.001, weight_decay=0.0001)

# 创建调度器
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 创建checkpoint目录
os.makedirs('./checkpoints', exist_ok=True)

# 保存测试checkpoint
print("保存测试checkpoint...")
torch.save({
    'epoch': 5,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_iou': 0.6543,
    'train_loss': 0.4321,
    'val_loss': 0.5678,
    'train_iou': 0.7654,
    'val_iou': 0.6543
}, './checkpoints/best_model.pth')

print("Checkpoint saved successfully.")
print("\n测试加载checkpoint...")

# 模拟train.py中的加载过程
checkpoint_path = './checkpoints/best_model.pth'
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # 加载训练统计信息
    start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
    best_iou = checkpoint['best_iou']
    
    print(f"Checkpoint loaded successfully.")
    print(f"  Starting from epoch: {start_epoch}")
    print(f"  Best IoU so far: {best_iou:.4f}")
else:
    print(f"No checkpoint found at {checkpoint_path}.")

# 清理测试文件
print("\n清理测试文件...")
os.remove('./checkpoints/best_model.pth')
os.rmdir('./checkpoints')

print("测试完成!")
