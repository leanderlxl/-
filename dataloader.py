import os
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import inspect
from typing import Callable, Optional, Dict, List, Tuple, Any

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils
except Exception:
    print("pycocotools not found. Please install it with 'pip install pycocotools'")
    raise




class SemanticSegTransform:
    """
    语义分割专用变换类，确保图像和掩码应用一致的几何变换
    
    Args:
        resize_size (tuple, optional): 调整大小尺寸 (h, w)
        flip_prob (float, optional): 水平翻转概率
        crop_prob (float, optional): 随机裁剪概率
        crop_size (tuple, optional): 随机裁剪尺寸 (h, w)
        color_jitter (dict, optional): 颜色增强参数 {brightness, contrast, saturation, hue}
        normalize (dict, optional): 归一化参数 {mean: [...], std: [...]}，默认为 ImageNet 归一化
    """
    def __init__(self, resize_size=None, flip_prob=0.5, crop_prob=0.5, crop_size=None, color_jitter=None, normalize=None):
        self.resize_size = resize_size
        self.flip_prob = flip_prob
        self.crop_prob = crop_prob
        self.crop_size = crop_size
        
        # 设置颜色增强
        if color_jitter is None:
            color_jitter = {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1}
        self.color_jitter = color_jitter
        
        # 默认使用 ImageNet 归一化
        if normalize is None:
            normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        self.normalize = normalize
    
    def __call__(self, image, mask):
        # 检查输入类型
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask)
        
        # 调整大小（如果需要）
        if self.resize_size is not None:
            # 对图像使用双线性插值
            image = image.resize(self.resize_size[::-1], Image.BILINEAR)
            # 对掩码使用最近邻插值
            mask = mask.resize(self.resize_size[::-1], Image.NEAREST)
        
        # 随机水平翻转
        if np.random.rand() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 随机裁剪
        if self.crop_size is not None and np.random.rand() < self.crop_prob:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
            image = transforms.functional.crop(image, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)
        
        # 颜色增强
        if self.color_jitter is not None:
            image = transforms.ColorJitter(
                brightness=self.color_jitter['brightness'],
                contrast=self.color_jitter['contrast'],
                saturation=self.color_jitter['saturation'],
                hue=self.color_jitter['hue']
            )(image)
        
        # 转换为张量
        image = transforms.ToTensor()(image)
        # 对图像进行归一化
        image = transforms.Normalize(mean=self.normalize['mean'], std=self.normalize['std'])(image)
        
        # 将mask转换为LongTensor，确保dtype和尺寸正确
        mask = np.array(mask, dtype=np.int64)
        
        # 严格验证mask中的值，确保只包含[0,80]和255
        unique_values = np.unique(mask)
        valid_values = set(range(81)) | {255}
        invalid_values = [val for val in unique_values if val not in valid_values]
        
        if invalid_values:
            print(f"Warning: Mask contains invalid values: {invalid_values}. Clipping to valid range...")
            # 将无效值裁剪到有效范围
            mask = np.clip(mask, 0, 80)
        
        mask = torch.as_tensor(mask, dtype=torch.long)
        
        return image, mask


class COCOSegDataset(Dataset):
    """
    COCO语义分割数据集类，在线生成掩码（不存储在硬盘上）
    
    Args:
        img_dir (str): 图像目录路径（如：'train2014' 或 'val2014'）
        ann_file (str): JSON 注释文件路径（如：'annotations/instances_train2014.json'）
        cat_map_path (str, optional): 类别映射文件路径（如：'cat_map.json'）
        transform (callable, optional): 对图像和掩码的变换函数，接收(图像, 掩码)并返回(变换后的图像, 变换后的掩码)
    """
    def __init__(self, img_dir: str, ann_file: str, cat_map_path: Optional[str] = None, transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None):
        self.img_dir: str = img_dir
        self.ann_file: str = ann_file
        self.transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = transform
        self.coco: COCO = COCO(ann_file)
        self.img_ids: List[int] = sorted(self.coco.getImgIds())
        self.cat_id_to_cont_id: Dict[int, int] = {}  # 初始化空字典
        self.cont_id_to_cat_id: Dict[int, int] = {}  # 初始化空字典
        self.cat_id_to_name: Dict[int, str] = {}  # 初始化空字典
        self.num_classes: int = 0  # 初始化默认值
        
        # 构建类别映射表
        if cat_map_path and os.path.exists(cat_map_path):
            # 如果提供了cat_map.json路径，则使用该文件中的映射
            with open(cat_map_path, 'r') as f:
                cat_map = json.load(f)
                # 将字符串键转换为整数
                self.cat_id_to_cont_id = {int(k): int(v) for k, v in cat_map['cat_id_to_cont'].items()}
                self.cont_id_to_cat_id = {int(k): int(v) for k, v in cat_map['cont_to_cat_id'].items()}
        else:
            # 如果没有提供cat_map.json路径，则手动构建映射
            categories = self.coco.dataset['categories']
            cat_ids = sorted([cat['id'] for cat in categories])
            self.cat_id_to_cont_id = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}  # 连续ID从1开始，背景=0
            self.cont_id_to_cat_id = {idx + 1: cat_id for idx, cat_id in enumerate(cat_ids)}
        
        # 构建类别ID到名称的映射
        categories = self.coco.dataset['categories']
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
        
        # 类别数量（包括背景）
        self.num_classes = len(self.cat_id_to_cont_id) + 1  # +1 表示背景
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            # 获取图像ID
            img_id = self.img_ids[idx]
            
            # 加载图像信息
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            # 检查图像是否存在
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                return None
            
            # 读取原图
            image = Image.open(img_path).convert('RGB')
            h, w = img_info['height'], img_info['width']
            
            # 验证图像尺寸是否与注释一致
            if image.height != h or image.width != w:
                print(f"Warning: Image size mismatch for image {img_id}: expected {w}x{h}, got {image.width}x{image.height}")
                return None
            
            # 创建空的语义分割掩码（背景=0）
            seg_mask = np.zeros((h, w), dtype=np.int64)  # 使用int64确保类别ID范围足够
            
            # 加载该图像的所有注释
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # 遍历所有注释，生成掩码
            for ann in anns:
                try:
                    # 检查是否为crowd区域，如果是则设置为ignore_index=255
                    iscrowd = ann.get('iscrowd', 0)
                    
                    if iscrowd == 1:
                        # crowd区域设置为ignore_index
                        binary_mask = self.coco.annToMask(ann)
                        if binary_mask.shape != (h, w):
                            binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        seg_mask[binary_mask == 1] = 255  # ignore_index
                    else:
                        # 正常注释
                        cat_id = ann['category_id']
                        if cat_id in self.cat_id_to_cont_id:
                            cont_id = self.cat_id_to_cont_id[cat_id]
                            # 确保类别ID在有效范围内 (0~80)
                            if cont_id > 80:
                                print(f"Warning: Category ID {cont_id} exceeds maximum allowed (80) for image {img_id}")
                                cont_id = min(cont_id, 80)
                            # 使用COCO API获取二值掩码
                            binary_mask = self.coco.annToMask(ann)  # 返回 HxW 的二值掩码（0/1）
                            
                            # 验证掩码尺寸是否与图像一致
                            if binary_mask.shape != (h, w):
                                binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            
                            # 按连续类别ID填充掩码
                            seg_mask[binary_mask == 1] = cont_id
                except Exception as e:
                    print(f"Warning: Error processing annotation {ann['id']} for image {img_id}: {e}")
                    continue
            
            # 转换为PIL Image以便应用transform
            seg_mask_img = Image.fromarray(seg_mask.astype(np.int32))  # PIL需要int32格式
            
            # 应用transform
            image_tensor = None
            mask_tensor = None
            
            if self.transform is not None:
                try:
                    # 同时对图像和掩码应用transform
                    transformed = self.transform(image, seg_mask_img)
                    
                    # 检查transform的返回值
                    if isinstance(transformed, tuple) and len(transformed) == 2:
                        image_result, mask_result = transformed
                        
                        # 确保图像是张量格式
                        if isinstance(image_result, torch.Tensor):
                            image_tensor = image_result
                        else:
                            # 使用ToTensor转换图像并归一化
                            image_tensor = transforms.ToTensor()(image_result)
                            image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
                        
                        # 确保掩码是numpy数组或PIL Image，然后转换为torch.long
                        if isinstance(mask_result, torch.Tensor):
                            # 验证mask值
                            mask_np = mask_result.cpu().numpy()
                            unique_values = np.unique(mask_np)
                            valid_values = set(range(81)) | {255}
                            invalid_values = [val for val in unique_values if val not in valid_values]
                            
                            if invalid_values:
                                print(f"Warning: Transform result - Mask contains invalid values: {invalid_values}. Clipping to valid range...")
                                mask_np = np.clip(mask_np, 0, 80)
                                mask_tensor = torch.as_tensor(mask_np, dtype=torch.long).to(mask_result.device)
                            else:
                                mask_tensor = mask_result.long()
                        else:
                            mask_np = np.array(mask_result, dtype=np.int64)
                            
                            # 验证mask值
                            unique_values = np.unique(mask_np)
                            valid_values = set(range(81)) | {255}
                            invalid_values = [val for val in unique_values if val not in valid_values]
                            
                            if invalid_values:
                                print(f"Warning: Transform result (non-tensor) - Mask contains invalid values: {invalid_values}. Clipping to valid range...")
                                mask_np = np.clip(mask_np, 0, 80)
                            
                            mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)
                    else:
                        print(f"Warning: Transform should return a tuple of (image, mask), got {type(transformed)}")
                        return None
                except Exception as e:
                    print(f"Error applying transform to image {img_id}: {e}")
                    # 应用默认变换
                    image_tensor = transforms.ToTensor()(image)
                    image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
                    
                    # 验证并转换mask
                    mask_np = np.array(seg_mask, dtype=np.int64)
                    unique_values = np.unique(mask_np)
                    valid_values = set(range(81)) | {255}
                    invalid_values = [val for val in unique_values if val not in valid_values]
                    
                    if invalid_values:
                        print(f"Warning: Exception handler - Mask contains invalid values: {invalid_values}. Clipping to valid range...")
                        mask_np = np.clip(mask_np, 0, 80)
                    
                    mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)
            else:
                # 应用默认变换
                image_tensor = transforms.ToTensor()(image)
                image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
                
                # 验证并转换mask
                mask_np = np.array(seg_mask, dtype=np.int64)
                unique_values = np.unique(mask_np)
                valid_values = set(range(81)) | {255}
                invalid_values = [val for val in unique_values if val not in valid_values]
                
                if invalid_values:
                    print(f"Warning: Default transform - Mask contains invalid values: {invalid_values}. Clipping to valid range...")
                    mask_np = np.clip(mask_np, 0, 80)
                
                mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)
            
            # 确保掩码是2D张量（HxW）
            if len(mask_tensor.shape) > 2:
                mask_tensor = mask_tensor.squeeze(0)
            
            # 验证图像和掩码尺寸一致
            if image_tensor.shape[1:] != mask_tensor.shape:
                print(f"Warning: Image and mask size mismatch for image {img_id}: image {image_tensor.shape[1:]}, mask {mask_tensor.shape}")
                # 调整掩码尺寸以匹配图像
                new_h, new_w = image_tensor.shape[1], image_tensor.shape[2]
                mask_np = mask_tensor.numpy()
                mask_np = cv2.resize(mask_np.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)
            
            return image_tensor, mask_tensor
        except Exception as e:
            print(f"Warning: Error in __getitem__ for index {idx}: {e}")
            return None


class COCO_SemanticSegDataset(Dataset):
    """
    COCO语义分割数据集类，在线生成掩码（不存储在硬盘上）
    
    Args:
        root_dir (str): 路径到 'train2014' 或 'val2014' 图像文件夹
        ann_file (str): annotations/instances_train2014.json 路径
        cat_map_path (str, optional): 类别映射文件路径（如：'cat_map.json'）
        transform: 对图像的变换
        target_transform: 对标签的变换
    """
    def __init__(self, root_dir: str, ann_file: str, cat_map_path: Optional[str] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        self.root_dir: str = root_dir
        self.ann_file: str = ann_file
        self.transform: Optional[Callable] = transform
        self.target_transform: Optional[Callable] = target_transform
        self.annotations: Dict[str, Any] = {}
        self.image_ids: List[int] = []
        self.cat_id_to_class: Dict[int, int] = {}  # 初始化空字典
        self.class_to_cat_id: Dict[int, int] = {}  # 初始化空字典
        self.class_to_name: Dict[int, str] = {}  # 初始化空字典
        self.num_classes: int = 0  # 初始化默认值
        self.all_annotations: List[Dict[str, Any]] = []

        # 加载 JSON 注释
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)

        # 获取图像列表
        self.image_ids = [img['id'] for img in self.annotations['images']]
        
        # 构建类别映射表
        if cat_map_path and os.path.exists(cat_map_path):
            # 如果提供了cat_map.json路径，则使用该文件中的映射
            with open(cat_map_path, 'r') as f:
                cat_map = json.load(f)
                # 将字符串键转换为整数
                self.cat_id_to_class = {int(k): int(v) for k, v in cat_map['cat_id_to_cont'].items()}
                self.class_to_cat_id = {int(k): int(v) for k, v in cat_map['cont_to_cat_id'].items()}
        else:
            # 如果没有提供cat_map.json路径，则手动构建映射
            # #map original COCO category_id (sparse, non-contiguous) to continuous class indices
            # reserve 0 for background
            self.cat_id_to_class = {cat['id']: idx + 1 for idx, cat in enumerate(self.annotations['categories'])}
            self.class_to_cat_id = {idx + 1: cat['id'] for idx, cat in enumerate(self.annotations['categories'])}
        
        self.class_to_name = {int(idx): cat['name'] for idx, cat in enumerate(self.annotations['categories'])}
        self.num_classes = len(self.cat_id_to_class) + 1  # +1 for background

        # 不再缓存所有图像的 mask 映射，改为动态加载
        # 只保存注释列表，在需要时根据图像ID过滤
        self.all_annotations = self.annotations['annotations']

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            img_id = self.image_ids[idx]
            # 动态获取数据集前缀，避免硬编码
            if 'train' in self.root_dir.lower():
                prefix = 'COCO_train2014'
            elif 'val' in self.root_dir.lower():
                prefix = 'COCO_val2014'
            else:
                prefix = 'COCO_train2014'  # 默认值
            
            # 使用COCO格式的文件名
            image_path = os.path.join(self.root_dir, f'{prefix}_{img_id:012d}.jpg')

            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                return None

            # 读取图像
            image = Image.open(image_path).convert("RGB")
            w, h = image.size

            # 创建空的语义分割图（全背景）
            seg_mask = np.zeros((h, w), dtype=np.int64)

            # 动态加载该图像的所有 annotation
            for ann in self.all_annotations:
                if ann['image_id'] == img_id:
                    if 'segmentation' not in ann:
                        continue
                    
                    # 检查是否为crowd区域
                    iscrowd = ann.get('iscrowd', 0)
                    
                    if iscrowd == 1:
                        # crowd区域设置为ignore_index
                        class_id = 255  # ignore_index
                    else:
                        # 正常注释
                        category_id = ann.get('category_id', None)
                        if category_id is None:
                            continue
                        class_id = self.cat_id_to_class.get(category_id, 0)
                    
                    # 确保类别ID在有效范围内 (0~80)
                    # 但保留ignore_index=255
                    if class_id != 255 and class_id > 80:
                        print(f"Warning: Category ID {class_id} exceeds maximum allowed (80) for image {img_id}")
                        class_id = min(class_id, 80)

                    # Prefer pycocotools decoding when available (handles polygons, lists of polygons, and RLE)
                    if maskUtils is not None:
                        try:
                            rle = ann['segmentation']
                            if isinstance(rle, dict):
                                mask_dec = maskUtils.decode(rle)
                            else:
                                # frPyObjects accepts list-of-polygons or list of RLEs
                                rles = maskUtils.frPyObjects(rle, h, w)
                                mask_dec = maskUtils.decode(rles)
                            if mask_dec.ndim == 3:
                                mask_bin = np.max(mask_dec, axis=2).astype(bool)
                            else:
                                mask_bin = mask_dec.astype(bool)
                            # 验证掩码尺寸是否与图像一致
                            if mask_bin.shape != (h, w):
                                mask_bin = cv2.resize(mask_bin.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                            seg_mask[mask_bin] = class_id
                            continue
                        except Exception as e:
                            print(f"Warning: Error decoding mask for annotation {ann['id']} using pycocotools: {e}")
                            # fallback to simple polygon handling below
                            pass

                    # Fallback: handle polygon list(s) without pycocotools
                    rle = ann['segmentation']
                    if isinstance(rle, list):
                        for polygon in rle:
                            if not polygon or len(polygon) % 2 != 0:
                                continue
                            mask_tmp = np.zeros((h, w), dtype=np.uint8)
                            try:
                                points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                                cv2.fillPoly(mask_tmp, [points], 1)
                                seg_mask[mask_tmp > 0] = class_id
                            except Exception as e:
                                print(f"Warning: Error processing polygon for annotation {ann['id']}: {e}")
                                continue

            # 转换为 PIL Image（使用 int32 保证 Image 支持整数类别）
            seg_mask_img = Image.fromarray(seg_mask.astype(np.int32))

            # 应用变换
            image_tensor = None
            mask_tensor = None
            
            if self.transform:
                # 检查transform是否接受两个参数（图像和掩码）
                sig = inspect.signature(self.transform.__call__)
                if len(sig.parameters) >= 2:
                    # 同时对图像和掩码应用相同的变换
                    transformed = self.transform(image, seg_mask_img)
                    # 检查transform的返回值
                    if isinstance(transformed, tuple) and len(transformed) == 2:
                        image_result, mask_result = transformed
                        
                        # 确保图像是张量格式
                        if isinstance(image_result, torch.Tensor):
                            image_tensor = image_result
                        else:
                            # 使用ToTensor转换图像并归一化
                            image_tensor = transforms.ToTensor()(image_result)
                            image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
                        
                        # 确保掩码是numpy数组或PIL Image，然后转换为torch.long
                        if isinstance(mask_result, torch.Tensor):
                            mask_tensor = mask_result.long()
                        else:
                            mask_np = np.array(mask_result, dtype=np.int64)
                            mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)
                    else:
                        print(f"Warning: Transform should return a tuple of (image, mask), got {type(transformed)}")
                        return None
                else:
                    # 只对图像应用变换
                    if isinstance(image, torch.Tensor):
                        image_tensor = image
                    else:
                        image_tensor = transforms.ToTensor()(image)
                        image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
                    
                    # 掩码不应用变换，保持原始尺寸
                    mask_tensor = torch.as_tensor(seg_mask, dtype=torch.long)
            else:
                # 应用默认变换
                image_tensor = transforms.ToTensor()(image)
                image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
                mask_tensor = torch.as_tensor(seg_mask, dtype=torch.long)

            if self.target_transform:
                # 确保target_transform不会使用ToTensor()转换掩码
                if hasattr(self.target_transform, '__name__') and self.target_transform.__name__ == 'ToTensor':
                    print(f"Warning: target_transform should not be ToTensor() for mask, skipping transform for image {img_id}")
                else:
                    try:
                        mask_tensor = self.target_transform(mask_tensor)
                    except Exception as e:
                        print(f"Warning: Error applying target_transform to image {img_id}: {e}")
                        return None

            # 确保掩码是2D张量（HxW）
            if len(mask_tensor.shape) > 2:
                mask_tensor = mask_tensor.squeeze(0)

            # 验证图像和掩码尺寸一致
            if image_tensor.shape[1:] != mask_tensor.shape:
                print(f"Warning: Image and mask size mismatch for image {img_id}: image {image_tensor.shape[1:]}, mask {mask_tensor.shape}")
                # 调整掩码尺寸以匹配图像
                new_h, new_w = image_tensor.shape[1], image_tensor.shape[2]
                mask_np = mask_tensor.numpy()
                mask_np = cv2.resize(mask_np.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)

            return image_tensor, mask_tensor
        except Exception as e:
            print(f"Warning: Error in __getitem__ for index {idx}: {e}")
            return None

    def _decode_rle(self, rle: Any, height: int, width: int) -> np.ndarray:
        """
        解码 COCO 的 RLE 编码 mask
        """
        # Prefer pycocotools if available
        if maskUtils is not None:
            if isinstance(rle, dict):
                return maskUtils.decode(rle)
            else:
                rles = maskUtils.frPyObjects(rle, height, width)
                mask = maskUtils.decode(rles)
                if mask.ndim == 3:
                    return np.max(mask, axis=2)
                return mask

        # Fallback simple decoder for list-format RLE (not full COCO RLE support)
        if isinstance(rle, list):
            mask = np.zeros(height * width, dtype=np.uint8)
            for i in range(0, len(rle), 2):
                start = rle[i]
                length = rle[i+1]
                mask[start:start+length] = 1
            return mask.reshape(height, width)
        raise RuntimeError("pycocotools not available and rle format not supported in fallback")