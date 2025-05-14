# 有监督CNN模型用于SDB检测

这个模块实现了使用有监督CNN直接进行SDB（睡眠呼吸障碍）检测的功能，作为SleepFM对比学习方法的基线对比。

## 原理

有监督CNN与对比学习模型的主要区别：

### 有监督CNN (传统方法)
1. **直接分类**：直接从原始信号学习分类SDB的特征
2. **损失函数**：使用交叉熵等分类损失函数
3. **数据要求**：需要大量带标签的数据
4. **学习过程**：直接学习输入信号与SDB标签之间的映射关系
5. **特点**：在训练数据充足的情况下表现较好，但对小样本场景不友好

### 对比学习模型 (SleepFM方法)
1. **两阶段学习**：
   - 预训练阶段：学习信号的通用表示(嵌入)，不需要SDB标签
   - 分类阶段：在嵌入上训练简单分类器(如逻辑回归)
2. **损失函数**：使用对比损失(如InfoNCE)，学习相似/不相似样本的表示
3. **数据利用**：可以利用无标签数据进行预训练
4. **特点**：能在较少标注数据的情况下取得良好性能

## 使用方法

### 数据准备

在运行模型之前，需要先准备SDB数据集。可以使用`prepare_sdb_dataset.py`脚本准备数据：

```bash
python prepare_sdb_dataset.py --data_dir /path/to/physionet_data --output_dir /path/to/output
```

### 训练模型

#### 单模态CNN

使用呼吸信号训练模型：

```bash
python train_supervised_cnn.py \
    --dataset_path /path/to/sdb_dataset.pkl \
    --model_type cnn \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-4
```

#### 多模态CNN

使用多种生理信号训练模型：

```bash
python train_supervised_cnn.py \
    --dataset_path /path/to/sdb_dataset.pkl \
    --model_type multimodal \
    --modalities Respiratory Sleep_Stages EKG \
    --fusion attention \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-4
```

### 参数说明

- `--dataset_path`: SDB数据集路径（pickle格式）
- `--output_dir`: 输出目录，保存模型和结果
- `--model_type`: 模型类型，可选`cnn`(单模态)或`multimodal`(多模态)
- `--fusion`: 多模态融合方法，可选`concat`、`attention`或`weighted`
- `--modalities`: 使用的模态，可选`Respiratory`、`Sleep_Stages`和`EKG`
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--weight_decay`: 权重衰减
- `--seed`: 随机种子
- `--gpu`: GPU ID，-1表示使用CPU

## 模型结构

### SupervisedCNN

单模态CNN模型结构:
- 5个卷积块，每个包含卷积层、批归一化、ReLU激活和最大池化
- 全局平均池化
- 3层全连接分类器，带Dropout正则化

### MultiModalCNN

多模态CNN模型结构:
- 为每个模态单独构建特征提取器
- 支持三种融合方法：拼接融合、注意力融合和权重融合
- 共享分类器头

## 结果输出

训练完成后，将在输出目录生成以下文件：
- `model.pth`: 训练好的模型权重
- `history.pkl`: 训练历史记录
- `results.pkl`: 测试集评估结果
- `loss_curve.png`: 损失曲线图
- `accuracy_curve.png`: 准确率曲线图
- `auc_auprc_curve.png`: AUC和AUPRC曲线图
- `confusion_matrix.png`: 混淆矩阵图
- `results.csv`: 评估指标汇总 