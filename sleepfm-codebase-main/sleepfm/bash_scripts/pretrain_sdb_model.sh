#!/bin/bash

# 设置参数
dataset_dir="/root/autodl-tmp/processed_data/pc18/sdb"
batch_size=32  # 恢复原来的批次大小
num_workers=4  # 恢复原来的工作线程数量
gradient_accumulation_steps=1  # 不使用梯度累积
weight_decay=0.01
lr=0.001
lr_step_period=5
epochs=20
mode="pairwise"  # 可选: pairwise, leave_one_out
modality_types="respiratory,sleep_stages,ekg"
respiratory_weight=1.3  # 呼吸信号权重
optimizer="adam"  # 可选: adam, sgd
adaptive_temp=true  # 是否使用自适应温度调整
initial_temp=0.07  # 初始温度值
min_temp=0.01  # 最小温度值
max_temp=0.5  # 最大温度值
temp_adjust_factor=0.05  # 温度调整因子
dynamic_weight=true  # 是否动态调整模态权重

# 输出路径在脚本内部自动生成
# 创建输出目录
# 输出目录在脚本内部自动创建

# 运行预训练脚本
cd .. && python 2_pretrain_sdb_model.py \
    --dataset_dir $dataset_dir \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay \
    --lr $lr \
    --lr_step_period $lr_step_period \
    --epochs $epochs \
    --mode $mode \
    --modality_types $modality_types \
    --respiratory_weight $respiratory_weight \
    --optimizer $optimizer \
    $([ "$adaptive_temp" = true ] && echo "--adaptive_temp") \
    --initial_temp $initial_temp \
    --min_temp $min_temp \
    --max_temp $max_temp \
    --temp_adjust_factor $temp_adjust_factor \
    $([ "$dynamic_weight" = true ] && echo "--dynamic_weight")

echo "SDB模型预训练完成！"
