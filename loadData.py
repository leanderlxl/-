import warnings
# 忽略torchvision.io的警告，因为我们使用PIL而不是torchvision.io来读取图像
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

from torchvision import transforms
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader import COCO_SemanticSegDataset
imagedir = '/Users/leanderlai/deepLearning/课设/dataset/train and annotation/train2014'
annfile = '/Users/leanderlai/deepLearning/课设/dataset/train and annotation/annotations/instances_train2014.json'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
])

# 初始化数据集
dataset = COCO_SemanticSegDataset(
    root_dir=imagedir,  # 图像路径
    ann_file=annfile,
    transform=transform,
    target_transform=target_transform
)
# 创建 DataLoader (batch_size=50 加载50张图片)
dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=0)

# 加载50张图片并查看效果
import matplotlib.pyplot as plt

# 获取一个batch的50张图片和对应的掩码
for images, masks in dataloader:
    print(f"加载的图片数量: {len(images)}")
    print(f"图片形状: {images.shape}")
    print(f"掩码形状: {masks.shape}")
    
    # 保存前5张图片及其掩码到本地文件
    import os
    output_dir = './output_images'
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(5):
        # 反归一化图片以便正确显示
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # 保存图片
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Image {i+1}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'image_{i+1}.png'), bbox_inches='tight')
        plt.close()
        
        # 保存掩码（使用 vmin/vmax 便于分类可视化）
        plt.figure(figsize=(8, 8))
        plt.imshow(masks[i].numpy(), cmap='jet', vmin=0, vmax=dataset.num_classes - 1)
        plt.title(f"Mask {i+1}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'mask_{i+1}.png'), bbox_inches='tight')
        plt.close()
    
    print(f"前5张图片和掩码已保存到 {output_dir} 目录")
    
    break