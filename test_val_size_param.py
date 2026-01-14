# 测试--val-size参数的功能

import sys
import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, Subset

# 添加当前目录到Python路径
sys.path.insert(0, os.getcwd())

# 创建一个模拟的验证数据集类
class MockValDataset(Dataset):
    def __init__(self):
        self.data = list(range(40504))  # 模拟COCO val2014的40504张图像
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 模拟parse_args函数
def parse_args_test(val_size_str):
    parser = argparse.ArgumentParser(description='Test val-size parameter')
    parser.add_argument('--val-size', type=str, default='all', help='Validation set size')
    
    # 使用提供的val_size_str参数
    sys.argv = ['test.py', f'--val-size={val_size_str}']
    args = parser.parse_args()
    return args

# 测试不同的val-size参数
def test_val_size(val_size_str, expected_size):
    print(f"\n=== 测试参数: --val-size={val_size_str} ===")
    
    # 创建模拟数据集
    val_dataset = MockValDataset()
    print(f"原始验证集大小: {len(val_dataset)}")
    
    # 解析参数
    args = parse_args_test(val_size_str)
    print(f"解析后的参数值: {args.val_size}")
    
    # 根据参数处理验证集
    if args.val_size.lower() != 'all':
        try:
            val_size = int(args.val_size)
            if val_size > 0 and val_size < len(val_dataset):
                val_indices = np.random.choice(len(val_dataset), val_size, replace=False)
                val_dataset = Subset(val_dataset, val_indices)
                print(f"使用{val_size}张验证集样本")
            else:
                print(f"验证集样本数{val_size}无效，使用全量验证集")
        except ValueError:
            print(f"验证集样本数参数{args.val_size}无效，使用全量验证集")
    else:
        print("使用全量验证集")
    
    # 检查结果
    actual_size = len(val_dataset)
    print(f"处理后的验证集大小: {actual_size}")
    
    if actual_size == expected_size or (expected_size == 'all' and actual_size == 40504):
        print("✅ 测试通过")
    else:
        print("❌ 测试失败")

# 运行测试
print("测试--val-size参数的功能")
print("=" * 50)

# 测试1: 使用全量验证集
test_val_size('all', 'all')

# 测试2: 使用指定数量的验证集
test_val_size('1000', 1000)

# 测试3: 使用无效的数字参数（过大）
test_val_size('50000', 'all')

# 测试4: 使用无效的数字参数（过小）
test_val_size('0', 'all')

# 测试5: 使用无效的非数字参数
test_val_size('abc', 'all')

print("\n" + "=" * 50)
print("所有测试完成！")
