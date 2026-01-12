from pycocotools.coco import COCO
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

# 设置路径
ann_file = '/Users/leanderlai/deepLearning/课设/dataset/train and annotation/annotations/instances_train2014.json'
img_dir = '/Users/leanderlai/deepLearning/课设/dataset/train and annotation/train2014'

# 初始化COCO API
coco = COCO(ann_file)

# 官方COCO类别列表（80类）
COCO_CATEGORIES = coco.dataset['categories']

# 构建映射：原始id → 连续id (1～80)
cat_ids = sorted([cat['id'] for cat in COCO_CATEGORIES])  # 长度80
cat_id_to_cont_id = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}  # 1～80
# 背景 = 0

# 获取图片ID列表
img_ids = coco.getImgIds()
print(f"总共有 {len(img_ids)} 张图片")

# 选择一张图片（这里选择第一张）
img_id = img_ids[0]

# 加载图片信息
img_info = coco.loadImgs(img_id)[0]
print(f"选择的图片信息: {img_info}")

# 加载图片
img_path = os.path.join(img_dir, f"COCO_train2014_{img_id:012d}.jpg")
print(f"图片路径: {img_path}")

# 检查图片是否存在
if not os.path.exists(img_path):
    print(f"图片不存在: {img_path}")
    # 尝试使用另一种命名格式
    img_path = os.path.join(img_dir, img_info['file_name'])
    print(f"尝试另一种路径: {img_path}")
    if not os.path.exists(img_path):
        print(f"图片仍然不存在: {img_path}")
        exit(1)

# 读取图片
image = Image.open(img_path).convert('RGB')
w, h = image.size

# 加载该图片的所有注释
ann_ids = coco.getAnnIds(imgIds=img_id)
anns = coco.loadAnns(ann_ids)
print(f"该图片有 {len(anns)} 个注释")

# 创建空的语义分割图（全背景）
seg_mask = np.zeros((h, w), dtype=np.uint8)  # 背景=0

# 遍历所有注释，生成掩码
for ann in anns:
    cat_id = ann['category_id']
    if cat_id in cat_id_to_cont_id:
        cont_id = cat_id_to_cont_id[cat_id]
        # 获取segmentation mask
        binary_mask = coco.annToMask(ann)  # 返回 HxW 的二值掩码
        # 按连续类别ID填充
        seg_mask[binary_mask == 1] = cont_id

# 保存图片和掩码到本地，以便查看
output_dir = './mask_output'
os.makedirs(output_dir, exist_ok=True)

# 保存原图
image.save(os.path.join(output_dir, f"image_{img_id}.png"))

# 保存掩码
mask_pil = Image.fromarray(seg_mask)
mask_pil.save(os.path.join(output_dir, f"mask_{img_id}.png"))

# 打印类别信息
print("\n该图片中的类别及其连续ID：")
unique_cont_ids = np.unique(seg_mask)
unique_cont_ids = unique_cont_ids[unique_cont_ids != 0]  # 排除背景
for cont_id in unique_cont_ids:
    # 找到对应的原始类别ID
    original_id = next(key for key, value in cat_id_to_cont_id.items() if value == cont_id)
    # 找到类别名称
    category_name = next(cat['name'] for cat in COCO_CATEGORIES if cat['id'] == original_id)
    print(f"连续ID: {cont_id}, 原始ID: {original_id}, 类别名称: {category_name}")

print(f"\n图片和掩码已保存到 {output_dir} 目录")
print(f"原图路径: {os.path.join(output_dir, f'image_{img_id}.png')}")
print(f"掩码路径: {os.path.join(output_dir, f'mask_{img_id}.png')}")

# 显示原图和掩码（用于验证）
print("\n正在显示原图和掩码...")

# 创建一个2x1的子图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 显示原图
axes[0].imshow(image)
axes[0].set_title(f"Original Image (ID: {img_id})")
axes[0].axis('off')

# 显示掩码
im = axes[1].imshow(seg_mask, cmap='jet')
axes[1].set_title(f"Segmentation Mask")
axes[1].axis('off')

# 添加颜色条
cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
cbar.set_label('Class ID')

# 保存显示结果
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"result_{img_id}.png"), bbox_inches='tight')
print(f"结果图已保存到: {os.path.join(output_dir, f'result_{img_id}.png')}")

# 在非交互式环境下，我们需要关闭plt.show()
# plt.show()