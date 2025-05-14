#!/bin/bash

# 激活conda环境
source activate sleepfm_env

# 设置参数
num_threads=4
dataset_dir="/root/autodl-tmp/processed_data/pc18/sdb"
random_state=42
# 数据集划分比例
pretrain_size=0.6  # 预训练集比例
train_size=0.2     # 训练集比例
valid_size=0.1     # 验证集比例
test_size=0.1      # 测试集比例
balance_ratio=3.0  # 设置为3.0表示非SDB样本是SDB样本的3倍，可以根据需要调整

# 运行SDB数据集划分脚本
python ../1_prepare_sdb_dataset.py \
    --dataset_dir $dataset_dir \
    --random_state $random_state \
    --pretrain_size $pretrain_size \
    --train_size $train_size \
    --valid_size $valid_size \
    --test_size $test_size \
    --num_threads $num_threads \
    --balance_ratio $balance_ratio

echo "SDB数据集划分完成!"
