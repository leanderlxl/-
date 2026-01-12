import numpy as np
from PIL import Image
import os

# 加载生成的掩码
mask_path = './mask_output/mask_57870.png'
mask = Image.open(mask_path)
mask_array = np.array(mask)

# 加载原始图片信息
image_path = './mask_output/image_57870.png'
image = Image.open(image_path)
image_array = np.array(image)

print(f"原始图片形状: {image_array.shape}")
print(f"掩码形状: {mask_array.shape}")
print(f"掩码数据类型: {mask_array.dtype}")

# 检查掩码中的唯一值（类别ID）
unique_values = np.unique(mask_array)
print(f"掩码中的唯一值: {unique_values}")

# 检查每个类别ID的像素数量
print("\n每个类别ID的像素数量:")
for value in unique_values:
    count = np.sum(mask_array == value)
    if value == 0:
        print(f"背景 (ID: {value}): {count} 像素")
    else:
        # 根据之前的结果，我们知道这些连续ID对应的类别
        id_to_class = {
            57: 'chair',
            59: 'potted plant',
            61: 'dining table',
            74: 'book',
            76: 'vase'
        }
        class_name = id_to_class.get(value, f'未知类别 (ID: {value})')
        print(f"{class_name} (连续ID: {value}): {count} 像素")

# 检查掩码是否与原图大小一致
if image_array.shape[:2] == mask_array.shape:
    print("\n✅ 掩码大小与原图大小一致")
else:
    print("\n❌ 掩码大小与原图大小不一致")

# 检查掩码是否只包含预期的类别ID
expected_ids = {0, 57, 59, 61, 74, 76}
actual_ids = set(unique_values)
if actual_ids.issubset(expected_ids):
    print("✅ 掩码只包含预期的类别ID")
else:
    unexpected_ids = actual_ids - expected_ids
    print(f"❌ 掩码包含意外的类别ID: {unexpected_ids}")

print("\n✅ 掩码验证完成！")
