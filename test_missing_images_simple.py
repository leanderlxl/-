# 测试代码处理缺失图像的能力（简化版，不依赖tensorboard）

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

# 创建临时目录和模拟图像
print("创建临时测试环境...")
test_dir = "./test_missing_images"
os.makedirs(test_dir, exist_ok=True)

# 创建一些模拟图像
for i in range(5):
    img = Image.new('RGB', (256, 256), color='red')
    img.save(os.path.join(test_dir, f"COCO_test_{i:012d}.jpg"))

# 创建简单的collate_fn
print("定义简单的collate_fn...")
def simple_collate_fn(batch):
    # 过滤掉None值
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()
    
    # 分离图像和掩码
    images, masks = zip(*batch)
    
    # 堆叠成批次
    images_tensor = torch.stack(images)
    masks_tensor = torch.stack(masks)
    
    return images_tensor, masks_tensor

# 创建一个简单的模拟数据集类
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

# 测试数据集的__getitem__方法
print("\n=== 测试数据集的__getitem__方法 ===")
dataset = MockCOCODataset()
success_count = 0
failure_count = 0

for i in range(len(dataset)):
    result = dataset[i]
    if result is not None:
        success_count += 1
        print(f"  索引 {i}: 成功加载")
    else:
        failure_count += 1
        print(f"  索引 {i}: 加载失败")

print(f"\n加载结果: 成功 {success_count} 个，失败 {failure_count} 个")

# 测试DataLoader和collate_fn
print("\n=== 测试DataLoader和collate_fn ===")
loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,
    collate_fn=simple_collate_fn
)

total_batches = 0
total_images_loaded = 0

for batch_idx, (images, masks) in enumerate(loader):
    batch_size = images.size(0) if images.numel() > 0 else 0
    total_images_loaded += batch_size
    total_batches += 1
    
    if batch_size > 0:
        print(f"  批次 {batch_idx}: 成功加载 {batch_size} 个图像")
        print(f"    图像形状: {images.shape}, 掩码形状: {masks.shape}")
    else:
        print(f"  批次 {batch_idx}: 没有成功加载的图像")

print(f"\nDataLoader 结果: 处理 {total_batches} 个批次，成功加载 {total_images_loaded} 个图像")

# 清理临时文件
print("\n清理临时测试环境...")
for i in range(5):
    os.remove(os.path.join(test_dir, f"COCO_test_{i:012d}.jpg"))
os.rmdir(test_dir)

print("\n测试完成!")
print("\n结论:")
print("1. 数据集类能够正确识别缺失的图像并返回None")
print("2. collate_fn能够正确过滤掉None值")
print("3. DataLoader能够处理包含缺失图像的数据集")
print("4. 整个流程能够优雅地处理缺失图像，不会中断程序")
