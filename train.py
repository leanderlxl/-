import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from typing import Union, Tuple, Optional
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt

from dataloader import COCOSegDataset, SemanticSegTransform

# 尝试导入torchmetrics，如果失败则使用手动计算
use_torchmetrics = True
try:
    from torchmetrics.classification import MulticlassJaccardIndex
except ImportError:
    use_torchmetrics = False
    print("Warning: torchmetrics not found, using manual mIoU calculation")

# 手动计算mIoU的函数
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
    
    # 计算非零类别的IoU
    valid_classes = (union > 0).sum()
    if valid_classes == 0:
        return 0.0
    
    iou_per_class = intersection[union > 0] / union[union > 0]
    mIoU = iou_per_class.mean().item()
    
    return mIoU

# 设置随机种子
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 解析命令行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training script for DeepLabv3 on COCO segmentation')
    parser.add_argument('--train-img-dir', type=str, default='./train2017', help='Training image directory')
    parser.add_argument('--train-ann-file', type=str, default='./annotations/instances_train2017.json', help='Training annotation file')
    parser.add_argument('--val-img-dir', type=str, default='./val2014', help='Validation image directory')
    parser.add_argument('--val-ann-file', type=str, default='./annotations/instances_val2014.json', help='Validation annotation file')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='steplr', choices=['steplr', 'cosine', 'plateau'], help='Scheduler type')
    parser.add_argument('--step-size', type=int, default=7, help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR scheduler')
    parser.add_argument('--resize-size', type=tuple, default=(256, 256), help='Resize size for images and masks')
    parser.add_argument('--flip-prob', type=float, default=0.5, help='Probability of horizontal flip')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone and only train classifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

# 主训练函数
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    
    print(f"Using device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    
    # 1. 设置模型
    print("\n1. Setting up model...")
    # 加载预训练的 DeepLabv3+ResNet50 模型
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    )
    
    # 替换分类头为 81 类（包括背景）
    num_classes = 81
    model.classifier[-1] = nn.Conv2d(
        in_channels=256,
        out_channels=num_classes,
        kernel_size=1
    )
    
    # 冻结 backbone
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    model = model.to(args.device)
    
    # 2. 设置数据加载器
    print("\n2. Setting up data loaders...")
    
    # 训练集变换
    train_transform = SemanticSegTransform(
        resize_size=args.resize_size,
        flip_prob=args.flip_prob
    )
    
    # 验证集变换（不需要翻转）
    val_transform = SemanticSegTransform(
        resize_size=args.resize_size,
        flip_prob=0.0
    )
    
    # 加载训练数据集
    train_dataset = COCOSegDataset(
        img_dir=args.train_img_dir,
        ann_file=args.train_ann_file,
        transform=train_transform
    )
    
    # 加载验证数据集
    val_dataset = COCOSegDataset(
        img_dir=args.val_img_dir,
        ann_file=args.val_ann_file,
        transform=val_transform
    )
    
    # 随机采样1000张验证集（固定seed确保可复现）
    val_indices = np.random.choice(len(val_dataset), 1000, replace=False)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # 自定义collate_fn，跳过错误图像
    def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        # 过滤掉None值（错误图像）
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            # 如果当前批次所有图像都出错，返回空批次
            return torch.Tensor(), torch.Tensor()
        # 使用默认的collate_fn处理过滤后的批次
        return torch.utils.data.dataloader.default_collate(batch)
    
    # 自定义Dataset包装器，用于处理异常
    class SafeDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __getitem__(self, idx: int) -> Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
            try:
                return self.dataset[idx]
            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                return None
        
        def __len__(self) -> int:
            return len(self.dataset)
    
    train_loader = DataLoader(
        SafeDataset(train_dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        SafeDataset(val_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # 3. 设置损失函数和优化器
    print("\n3. Setting up loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    
    # 只优化需要梯度的参数
    params_to_optimize = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 根据选择创建不同的调度器
    if args.scheduler == 'steplr':
        scheduler = StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # 最大化mIoU
            factor=args.gamma,
            patience=3,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {args.scheduler}")
    
    # 4. 加载checkpoint（如果存在）
    print("\n4. Checking for checkpoint...")
    start_epoch = 0
    best_iou = 0.0
    checkpoint_path = './checkpoints/best_model.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        
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
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
    
    # 5. 设置评估指标 (mIoU)
    print("\n5. Setting up evaluation metrics...")
    iou_metric = None
    if use_torchmetrics:
        iou_metric = MulticlassJaccardIndex(num_classes=num_classes, average='macro').to(args.device)
    
    # 6. 初始化TensorBoard
    print("\n6. Setting up TensorBoard...")
    tensorboard_dir = '/root/tf-logs'
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # 7. 训练循环
    print("\n7. Starting training loop...")
    early_stopping_patience = 5  # 设置早停耐心值
    epochs_without_improvement = 0  # 跟踪没有改进的轮数
    
    for epoch in range(start_epoch, args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training") as pbar:
            for images, masks in pbar:
                # 跳过空批次
                if images.size(0) == 0:
                    continue
                    
                images = images.to(args.device)
                masks = masks.to(args.device)
                
                # 前向传播
                outputs = model(images)
                out = outputs['out']
                
                # 计算损失
                loss = criterion(out, masks)
                
                # 梯度更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 计算 IoU
                with torch.no_grad():
                    predicted = out.argmax(dim=1)
                    if use_torchmetrics and iou_metric is not None:
                        batch_iou = iou_metric(predicted, masks)
                    else:
                        batch_iou = calculate_miou(predicted, masks, num_classes)
                    
                # 更新统计信息（使用batch平均）
                train_loss += loss.item()
                train_iou += batch_iou.item() if isinstance(batch_iou, torch.Tensor) else batch_iou
                train_batches += 1
                
                # 格式化IoU值，根据类型决定是否调用.item()
                iou_value = batch_iou.item() if isinstance(batch_iou, torch.Tensor) else batch_iou
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou_value:.4f}'
                })
        
        # 计算 epoch 训练指标
        if train_batches > 0:
            train_loss /= train_batches
            train_iou /= train_batches
        else:
            train_loss = 0.0
            train_iou = 0.0
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_batches = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation") as pbar:
                for images, masks in pbar:
                    # 跳过空批次
                    if images.size(0) == 0:
                        continue
                    
                    images = images.to(args.device)
                    masks = masks.to(args.device)
                    
                    # 前向传播
                    outputs = model(images)
                    out = outputs['out']
                    
                    # 计算损失
                    loss = criterion(out, masks)
                    
                    # 计算 IoU
                    predicted = out.argmax(dim=1)
                    if use_torchmetrics and iou_metric is not None:
                        batch_iou = iou_metric(predicted, masks)
                    else:
                        batch_iou = calculate_miou(predicted, masks, num_classes)
                    
                    # 可视化预测结果，每隔2个epoch记录一次
                    if val_batches == 0 and (epoch % 2 == 0):  # 每个epoch的第一个batch，每隔2个epoch记录一次
                        # 选择第一张图片进行可视化
                        
                        # 恢复图像的原始颜色
                        inv_normalize = transforms.Compose([
                            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
                        ])
                        
                        img = inv_normalize(images[0]).cpu().numpy().transpose(1, 2, 0)
                        img = np.clip(img, 0, 1)
                        
                        mask = masks[0].cpu().numpy()
                        pred = predicted[0].cpu().numpy()
                        
                        # 创建可视化图像
                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                        axs[0].imshow(img)
                        axs[0].set_title('Image')
                        axs[0].axis('off')
                        
                        axs[1].imshow(mask, cmap='jet')
                        axs[1].set_title('Ground Truth')
                        axs[1].axis('off')
                        
                        axs[2].imshow(pred, cmap='jet')
                        axs[2].set_title('Prediction')
                        axs[2].axis('off')
                        
                        plt.tight_layout()
                        
                        # 将图像写入TensorBoard
                        writer.add_figure('Val/Prediction_Example', fig, epoch)
                        plt.close(fig)  # 显式关闭figure，释放内存
                    
                    # 递增batch计数
                    val_batches += 1
                    
                    # 更新统计信息（使用batch平均）
                    val_loss += loss.item()
                    val_iou += batch_iou.item() if isinstance(batch_iou, torch.Tensor) else batch_iou
                    
                    # 格式化IoU值，根据类型决定是否调用.item()
                    iou_value = batch_iou.item() if isinstance(batch_iou, torch.Tensor) else batch_iou
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'iou': f'{iou_value:.4f}'
                    })
        
        # 计算 epoch 验证指标
        if val_batches > 0:
            val_loss /= val_batches
            val_iou /= val_batches
        else: 
            val_loss = 0.0
            val_iou = 0.0
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('mIoU/Train', train_iou, epoch)
        writer.add_scalar('mIoU/Val', val_iou, epoch)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Params/Learning_Rate', current_lr, epoch)
        
        # 强制刷新到硬盘，防止程序崩溃导致数据丢失
        writer.flush()
        
        # 更新学习率
        if args.scheduler == 'plateau':
            scheduler.step(val_iou)  # 传递验证IoU给ReduceLROnPlateau
        else:
            scheduler.step()  # StepLR和CosineAnnealingLR不需要参数
        
        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_iou': train_iou,
                'val_iou': val_iou
            }, './checkpoints/best_model.pth')
            print(f"New best model saved with IoU: {best_iou:.4f}")
            # 重置没有改进的轮数
            epochs_without_improvement = 0
        else:
            # 没有改进，增加计数
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        # 检查早停条件
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs without improvement")
            break
        
        # 打印 epoch 结果
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        print(f"  Best Val IoU: {best_iou:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print()
    
    # 关闭TensorBoard writer
    writer.close()
    
    print("\nTraining completed!")
    print(f"Best validation IoU: {best_iou:.4f}")

if __name__ == '__main__':
    main()
