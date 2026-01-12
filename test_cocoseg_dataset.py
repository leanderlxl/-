import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from dataloader import COCOSegDataset

# 设置路径
img_dir = '/Users/leanderlai/deepLearning/课设/dataset/train and annotation/train2014'
ann_file = '/Users/leanderlai/deepLearning/课设/dataset/train and annotation/annotations/instances_train2014.json'

# 创建一个简单的transform，用于测试
class SimpleTransform:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, mask):
        # 对图像应用transform（使用默认插值）
        image_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])
        
        # 对掩码应用transform（必须使用INTER_NEAREST插值）
        mask_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST),
        ])
        
        return image_transform(image), mask_transform(mask)

# 测试函数
def test_coco_seg_dataset():
    print("正在测试COCOSegDataset类...")
    
    # 初始化数据集（无transform）
    dataset = COCOSegDataset(img_dir=img_dir, ann_file=ann_file)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数量（包括背景）: {dataset.num_classes}")
    
    # 测试__getitem__方法
    idx = 0
    image_tensor, mask_tensor = dataset.__getitem__(idx)
    
    print(f"\n测试第 {idx} 个样本:")
    print(f"图像形状: {image_tensor.shape}")
    print(f"掩码形状: {mask_tensor.shape}")
    print(f"图像数据类型: {image_tensor.dtype}")
    print(f"掩码数据类型: {mask_tensor.dtype}")
    
    # 检查掩码中的唯一值
    mask_array = mask_tensor.numpy()
    unique_values = np.unique(mask_array)
    print(f"掩码中的唯一值: {unique_values}")
    
    # 检查类别映射
    print("\n类别映射示例:")
    for i, (cat_id, cont_id) in enumerate(list(dataset.cat_id_to_cont_id.items())[:5]):
        cat_name = dataset.cat_id_to_name[cat_id]
        print(f"  原始ID: {cat_id} → 连续ID: {cont_id} → 类别名称: {cat_name}")
    
    # 测试带有transform的数据集
    print("\n\n正在测试带有transform的数据集...")
    transform = SimpleTransform((512, 512))
    dataset_with_transform = COCOSegDataset(img_dir=img_dir, ann_file=ann_file, transform=transform)
    
    # 测试__getitem__方法
    idx = 0
    image_tensor, mask_tensor = dataset_with_transform.__getitem__(idx)
    
    print(f"\n测试第 {idx} 个样本（带transform）:")
    print(f"图像形状: {image_tensor.shape}")
    print(f"掩码形状: {mask_tensor.shape}")
    print(f"图像数据类型: {image_tensor.dtype}")
    print(f"掩码数据类型: {mask_tensor.dtype}")
    
    # 检查掩码中的唯一值
    mask_array = mask_tensor.numpy()
    unique_values = np.unique(mask_array)
    print(f"掩码中的唯一值: {unique_values}")
    
    # 保存结果用于可视化
    output_dir = './dataset_test_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始图像（不带transform）
    original_image, original_mask = dataset.__getitem__(idx)
    original_image_pil = transforms.ToPILImage()(original_image)
    original_image_pil.save(os.path.join(output_dir, 'original_image.png'))
    
    # 保存掩码（不带transform）
    original_mask_pil = Image.fromarray(original_mask.numpy().astype(np.uint8))
    original_mask_pil.save(os.path.join(output_dir, 'original_mask.png'))
    
    # 保存transform后的图像
    transformed_image_pil = transforms.ToPILImage()(image_tensor)
    transformed_image_pil.save(os.path.join(output_dir, 'transformed_image.png'))
    
    # 保存transform后的掩码
    transformed_mask_pil = Image.fromarray(mask_tensor.numpy().astype(np.uint8))
    transformed_mask_pil.save(os.path.join(output_dir, 'transformed_mask.png'))
    
    print(f"\n结果已保存到 {output_dir} 目录")
    print("\n✅ COCOSegDataset类测试完成！")

if __name__ == "__main__":
    test_coco_seg_dataset()
