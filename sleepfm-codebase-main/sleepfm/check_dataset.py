#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查SDB数据集格式
"""

import pickle
import sys
from pprint import pprint

def check_dataset(dataset_path):
    """检查数据集格式"""
    print(f"检查数据集: {dataset_path}")
    
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print("数据集类型:", type(dataset))
        
        if isinstance(dataset, dict):
            print("数据集键:", dataset.keys())
            
            for key in dataset:
                print(f"\n{key} 分割:")
                if isinstance(dataset[key], dict):
                    print(f"  {key} 分割类型: dict")
                    print(f"  {key} 分割键:", dataset[key].keys())
                    
                    for subkey in dataset[key]:
                        value = dataset[key][subkey]
                        value_type = type(value)
                        value_len = len(value) if hasattr(value, '__len__') else 'N/A'
                        print(f"    {subkey}: 类型={value_type}, 长度={value_len}")
                        
                        # 如果是列表或数组，显示第一个元素
                        if hasattr(value, '__len__') and len(value) > 0:
                            print(f"      第一个元素: {value[0]}")
                else:
                    print(f"  {key} 分割类型:", type(dataset[key]))
                    if hasattr(dataset[key], '__len__'):
                        print(f"  {key} 分割长度:", len(dataset[key]))
                        
                        # 如果是列表或数组，显示第一个元素
                        if len(dataset[key]) > 0:
                            print(f"  第一个元素: {dataset[key][0]}")
        else:
            print("数据集不是字典类型")
            
    except Exception as e:
        print(f"检查数据集时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "/root/autodl-tmp/processed_data/pc18/sdb/sdb_dataset_paths.pickle"
    
    check_dataset(dataset_path)
