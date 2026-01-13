# 测试代码处理缺失图像的能力

import os
import torch
import numpy as np
from PIL import Image
from dataloader import COCOSegDataset, COCO_SemanticSegDataset
from train import SafeDataset, collate_fn
from torch.utils.data import DataLoader

# 创建临时目录和模拟图像
print("创建临时测试环境...")
test_dir = "./test_missing_images"
os.makedirs(test_dir, exist_ok=True)

# 创建一些模拟图像
for i in range(5):
    img = Image.new('RGB', (256, 256), color='red')
    img.save(os.path.join(test_dir, f"COCO_test_{i:012d}.jpg"))

# 创建一个简单的模拟COCO数据集类
class MockCOCODataset:
    def __init__(self):
        self.image_ids = list(range(10))  # 10个图像ID，其中5个存在，5个不存在
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(test_dir, f"COCO_test_{img_id:012d}.jpg")
        
        print(f"尝试加载图像 {img_id}，路径: {img_path}")
        
        # 检查图像是否存在
        if not os.path.exists(img_path):
            print(f"Warning: 图像 {img_id} 不存在")
            return None
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        
        # 创建简单的掩码
        mask = np.zeros((256, 256), dtype=np.int64)
        mask[100:150, 100:150] = 1  # 简单的矩形掩码
        
        # 转换为张量
        image_tensor = torch.from_numpy(np.array(image).transpose((2, 0, 1))).float() / 255.0
        mask_tensor = torch.from_numpy(mask).long()
        
        return image_tensor, mask_tensor

# 测试直接使用MockCOCODataset
print("\n=== 测试MockCOCODataset的__getitem__方法 ===")
dataset = MockCOCODataset()
for i in range(len(dataset)):
    result = dataset[i]
    print(f"  索引 {i}: {'成功加载' if result is not None else '加载失败'}")

# 测试使用SafeDataset包装
print("\n=== 测试SafeDataset包装器 ===")
safe_dataset = SafeDataset(dataset)
for i in range(len(safe_dataset)):
    result = safe_dataset[i]
    print(f"  索引 {i}: {'成功加载' if result is not None else '加载失败'}")

# 测试DataLoader和collate_fn
print("\n=== 测试DataLoader和collate_fn ===")
loader = DataLoader(
    safe_dataset,
    batch_size=3,
    shuffle=True,
    collate_fn=collate_fn
)

total_batches = 0
total_images = 0

for batch_idx, (images, masks) in enumerate(loader):
    print(f"  批次 {batch_idx}: 图像形状 {images.shape if images.numel() > 0 else '空'}, 掩码形状 {masks.shape if masks.numel() > 0 else '空'}")
    
    if images.numel() > 0:
        total_batches += 1
        total_images += images.size(0)

print(f"\n总结: 共处理 {total_batches} 个批次，{total_images} 个图像")
print("预期: 应该跳过不存在的图像，只加载存在的5个图像")

# 清理临时文件
print("\n清理临时测试环境...")
for i in range(5):
    os.remove(os.path.join(test_dir, f"COCO_test_{i:012d}.jpg"))
os.rmdir(test_dir)

print("测试完成!")
