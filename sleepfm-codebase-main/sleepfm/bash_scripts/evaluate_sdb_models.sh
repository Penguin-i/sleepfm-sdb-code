#!/bin/bash

# SDB检测模型评估脚本
# 该脚本用于评估不同模态的SDB检测模型性能

# 设置参数
dataset_dir="/root/autodl-tmp/processed_data/pc18/sdb"
output_dir="outputs/sdb_pretrain_leave_one_out_sdb_dataset_paths_lr_0.001_lr_sp_5_wd_0.01_bs_32_respiratory_sleep_stages_ekg_resp_w1.3"
model_name="logistic"  # 可选: logistic, xgb, rf
max_iter=1000

# 创建输出目录
mkdir -p "$dataset_dir/$output_dir/evaluation"

# 评估单模态模型
echo "评估呼吸信号模型..."
python sdb_classification_eval.py \
    --output_file "$output_dir" \
    --dataset_dir "$dataset_dir" \
    --modality_type "respiratory" \
    --model_name "$model_name" \
    --max_iter "$max_iter"

echo "评估睡眠阶段模型..."
python sdb_classification_eval.py \
    --output_file "$output_dir" \
    --dataset_dir "$dataset_dir" \
    --modality_type "sleep_stages" \
    --model_name "$model_name" \
    --max_iter "$max_iter"

echo "评估心电图模型..."
python sdb_classification_eval.py \
    --output_file "$output_dir" \
    --dataset_dir "$dataset_dir" \
    --modality_type "ekg" \
    --model_name "$model_name" \
    --max_iter "$max_iter"

# 评估多模态融合模型
echo "评估多模态融合模型..."
python sdb_classification_eval.py \
    --output_file "$output_dir" \
    --dataset_dir "$dataset_dir" \
    --modality_type "combined" \
    --model_name "$model_name" \
    --max_iter "$max_iter"

echo "SDB检测模型评估完成!"
