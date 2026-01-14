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
def update_miou_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, 
                        intersection: torch.Tensor, union: torch.Tensor, ignore_label: int = 255) -> None:
    """
    更新多类IoU计算的累积交集和并集
    
    Args:
        preds: 预测结果，形状为[B, H, W]的long张量
        targets: 真实标签，形状为[B, H, W]的long张量
        num_classes: 类别数量
        intersection: 累积交集张量，形状为[num_classes]
        union: 累积并集张量，形状为[num_classes]
        ignore_label: 要忽略的标签值
    """
    # 过滤掉ignore标签的像素
    valid_mask = (targets != ignore_label)
    preds_valid = preds[valid_mask]
    targets_valid = targets[valid_mask]
    
    for cls in range(1, num_classes):  # 忽略背景类0
        pred_cls = (preds_valid == cls)
        target_cls = (targets_valid == cls)
        
        cls_intersection = (pred_cls & target_cls).sum()
        cls_union = (pred_cls | target_cls).sum()
        
        intersection[cls] += cls_intersection
        union[cls] += cls_union


def calculate_miou_from_metrics(intersection: torch.Tensor, union: torch.Tensor, num_classes: int) -> float:
    """
    从累积的交集和并集计算最终的mIoU
    
    Args:
        intersection: 累积交集张量，形状为[num_classes]
        union: 累积并集张量，形状为[num_classes]
        num_classes: 类别数量
        
    Returns:
        mIoU: 平均IoU值
    """
    # 计算所有前景类别的IoU，未出现的类别IoU视为0
    foreground_classes = range(1, num_classes)  # 1到num_classes-1，不包括背景
    
    iou_per_class = []
    for cls in foreground_classes:
        if union[cls] > 0:
            iou_per_class.append(intersection[cls] / union[cls])
        else:
            iou_per_class.append(torch.tensor(0.0, device=intersection.device))
    
    mIoU = torch.stack(iou_per_class).mean().item()
    
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
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['steplr', 'cosine', 'plateau'], help='Scheduler type')
    parser.add_argument('--step-size', type=int, default=7, help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma for StepLR scheduler')
    parser.add_argument('--resize-size', type=tuple, default=(512, 512), help='Resize size for images and masks')
    parser.add_argument('--flip-prob', type=float, default=0.5, help='Probability of horizontal flip')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone and only train classifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--val-size', type=str, default='all', help='Validation set size (use "all" for full dataset, or an integer for sample size)')
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
    # 使用weights_backbone参数只加载ImageNet backbone预训练权重
    # segmentation head不加载预训练权重（随机初始化）
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None,  # 不加载完整模型权重
        weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V1  # 只加载backbone的ImageNet权重
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
    
    # 验证集变换（不需要任何随机行为）
    val_transform = SemanticSegTransform(
        resize_size=args.resize_size,
        flip_prob=0.0,
        crop_prob=0.0,  # 禁用随机裁剪
        color_jitter=None  # 禁用颜色增强
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
    
    # 根据参数决定使用全量验证集还是样本集
    if args.val_size.lower() != 'all':
        try:
            val_size = int(args.val_size)
            if val_size > 0 and val_size < len(val_dataset):
                val_indices = np.random.choice(len(val_dataset), val_size, replace=False)
                val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
                print(f"使用{val_size}张验证集样本")
            else:
                print(f"验证集样本数{val_size}无效，使用全量验证集")
        except ValueError:
            print(f"验证集样本数参数{args.val_size}无效，使用全量验证集")
    else:
        print("使用全量验证集")
    
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
    # 使用CrossEntropyLoss并设置ignore_index来忽略无效像素（255）
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # 只优化需要梯度的参数
    params_to_optimize = [param for param in model.parameters() if param.requires_grad]
    # 使用AdamW优化器，在语义分割+weight_decay场景下比Adam更稳定
    optimizer = optim.AdamW(
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
            patience=5,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {args.scheduler}")
    
    # 4. 初始化训练状态（不加载旧checkpoint，从epoch 0开始）
    print("\n4. Initializing training state...")
    start_epoch = 0
    best_iou = 0.0
    checkpoint_path = './checkpoints/best_model.pth'
    
    # 根据改进指南，我们从epoch 0开始重新训练，不加载旧模型
    print("Starting fresh training from epoch 0")
    print("(Note: Not loading any previous checkpoint as per the improvement guidelines)")
    
    # 5. 设置评估指标 (mIoU)
    print("\n5. Setting up evaluation metrics...")
    # 训练和验证分别使用独立的IoU指标实例，避免互相污染
    # 注意：ignore_index=0 表示忽略背景类，与手写IoU的逻辑保持一致（只计算前景类）
    train_iou_metric = None
    val_iou_metric = None
    if use_torchmetrics:
        train_iou_metric = MulticlassJaccardIndex(
            num_classes=num_classes, 
            average='macro',
            ignore_index=0  # 忽略背景类
        ).to(args.device)
        val_iou_metric = MulticlassJaccardIndex(
            num_classes=num_classes, 
            average='macro',
            ignore_index=0  # 忽略背景类
        ).to(args.device)
    
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
        train_batches = 0
        
        # 初始化累积交集和并集
        train_intersection = torch.zeros(num_classes, device=args.device)
        train_union = torch.zeros(num_classes, device=args.device)
        
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
                
                # 计算损失，忽略背景类0
                loss = criterion(out, masks)
                
                # 梯度更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 更新统计信息
                train_loss += loss.item()
                train_batches += 1
                
                # 计算并累积IoU指标
                with torch.no_grad():
                    predicted = out.argmax(dim=1)
                    
                    # 确保predicted值在有效范围内
                    pred_min = predicted.min().item()
                    pred_max = predicted.max().item()
                    
                    if pred_max >= num_classes:
                        predicted = torch.clamp(predicted, 0, num_classes - 1)
                        pred_max = num_classes - 1
                    

                    # 处理crowd像素（值为255），将其转换为ignore_index=0
                    masks_processed = masks.clone()
                    masks_processed[masks_processed == 255] = 0  # 将crowd像素转换为背景
                    
                    # 确保masks_processed的值只在0-80范围内
                    masks_processed = torch.clamp(masks_processed, 0, 80)
                    
                    if use_torchmetrics and train_iou_metric is not None:
                        train_iou_metric(predicted, masks_processed)
                    else:
                        update_miou_metrics(predicted, masks_processed, num_classes, train_intersection, train_union)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })
        
        # 计算 epoch 训练指标
        if train_batches > 0:
            train_loss /= train_batches
            if use_torchmetrics and train_iou_metric is not None:
                train_iou = train_iou_metric.compute().item()
                train_iou_metric.reset()
            else:
                train_iou = calculate_miou_from_metrics(train_intersection, train_union, num_classes)
        else:
            train_loss = 0.0
            train_iou = 0.0
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        # 初始化累积交集和并集
        val_intersection = torch.zeros(num_classes, device=args.device)
        val_union = torch.zeros(num_classes, device=args.device)
        
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
                    
                    # 保存当前批次的统计信息用于可视化
                    current_batch_loss = loss.item()
                    current_batch_images = images
                    current_batch_masks = masks
                    
                    # 更新统计信息
                    val_loss += current_batch_loss
                    val_batches += 1
                    
                    # 计算并累积IoU指标
                    predicted = out.argmax(dim=1)
                    

                    
                    # 处理crowd像素（值为255），将其转换为ignore_index=0
                    masks_processed = masks.clone()
                    masks_processed[masks_processed == 255] = 0  # 将crowd像素转换为背景
                    
                    # 确保masks_processed的值只在0-80范围内
                    masks_processed = torch.clamp(masks_processed, 0, 80)
                    
                    if use_torchmetrics and val_iou_metric is not None:
                        val_iou_metric(predicted, masks_processed)
                    else:
                        update_miou_metrics(predicted, masks_processed, num_classes, val_intersection, val_union)
                    
                    # 可视化预测结果，每隔2个epoch记录一次
                    # 这里val_batches已经递增，但我们检查val_batches == 1表示第一个批次
                    if val_batches == 1 and (epoch % 2 == 0):  # 每个epoch的第一个batch，每隔2个epoch记录一次
                        # 选择第一张图片进行可视化
                        
                        # 恢复图像的原始颜色
                        inv_normalize = transforms.Compose([
                            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
                        ])
                        
                        img = inv_normalize(current_batch_images[0]).cpu().numpy().transpose(1, 2, 0)
                        img = np.clip(img, 0, 1)
                        
                        mask = current_batch_masks[0].cpu().numpy()
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
                    
                    # 设置进度条显示
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}'
                    })
        
        # 计算 epoch 验证指标
        if val_batches > 0:
            val_loss /= val_batches
            if use_torchmetrics and val_iou_metric is not None:
                val_iou = val_iou_metric.compute().item()
                val_iou_metric.reset()
            else:
                val_iou = calculate_miou_from_metrics(val_intersection, val_union, num_classes)
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
