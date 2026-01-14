# 测试calculate_miou函数的修改

import torch

# 直接复制calculate_miou函数的代码
def calculate_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    计算多类IoU的平均值
    
    Args:
        preds: 预测结果，形状为[B, H, W]的long张量
        targets: 真实标签，形状为[B, H, W]的long张量
        num_classes: 类别数量
        
    Returns:
        mIoU: 平均IoU值
    """
    # 计算每个类别和批次的交集和并集    
    intersection = torch.zeros(num_classes, device=preds.device)
    union = torch.zeros(num_classes, device=preds.device)
    
    for cls in range(num_classes):
        # 忽略背景类0
        if cls == 0:
            continue
            
        # 计算当前类别的交集和并集
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        cls_intersection = (pred_cls & target_cls).sum()
        cls_union = (pred_cls | target_cls).sum()
        
        # 避免除以零
        if cls_union > 0:
            intersection[cls] = cls_intersection
            union[cls] = cls_union
    
    # 计算所有前景类别的IoU，未出现的类别IoU视为0
    foreground_classes = range(1, num_classes)  # 1到num_classes-1，不包括背景
    
    iou_per_class = []
    for cls in foreground_classes:
        if union[cls] > 0:
            iou_per_class.append(intersection[cls] / union[cls])
        else:
            iou_per_class.append(torch.tensor(0.0, device=preds.device))
    
    mIoU = torch.stack(iou_per_class).mean().item()
    
    return mIoU

# 测试calculate_miou函数
print('测试calculate_miou函数...')

# 创建模拟数据
num_classes = 5  # 小测试，包括背景类
# 模拟预测结果，形状为[B, H, W]
preds = torch.randint(0, num_classes, (2, 10, 10))
# 模拟真实标签，形状为[B, H, W]
targets = torch.randint(0, num_classes, (2, 10, 10))

# 确保某些类别在数据中不出现
targets[targets == 2] = 0  # 移除类别2
targets[targets == 3] = 0  # 移除类别3

# 计算mIoU
mIoU = calculate_miou(preds, targets, num_classes)
print(f'mIoU: {mIoU:.4f}')
print('测试完成！')
