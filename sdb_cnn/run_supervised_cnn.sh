#!/bin/bash

# 运行有监督CNN模型训练SDB检测器的脚本

# 设置数据路径
DATASET_PATH="/root/autodl-tmp/processed_data/sdb_dataset.pkl"
OUTPUT_DIR="./models/supervised_cnn"

# 单模态CNN - 呼吸信号
echo "Running respiratory modality CNN..."
python train_supervised_cnn.py \
    --dataset_path $DATASET_PATH \
    --output_dir "${OUTPUT_DIR}/respiratory" \
    --model_type cnn \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --seed 42

# 多模态CNN - 所有模态拼接融合
echo "Running multimodal CNN with concatenation fusion..."
python train_supervised_cnn.py \
    --dataset_path $DATASET_PATH \
    --output_dir "${OUTPUT_DIR}/multimodal_concat" \
    --model_type multimodal \
    --fusion concat \
    --modalities Respiratory Sleep_Stages EKG \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --seed 42

# 多模态CNN - 所有模态注意力融合
echo "Running multimodal CNN with attention fusion..."
python train_supervised_cnn.py \
    --dataset_path $DATASET_PATH \
    --output_dir "${OUTPUT_DIR}/multimodal_attention" \
    --model_type multimodal \
    --fusion attention \
    --modalities Respiratory Sleep_Stages EKG \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --seed 42

echo "All models trained successfully!" 