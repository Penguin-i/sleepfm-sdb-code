#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
有监督CNN模型用于SDB（睡眠呼吸障碍）检测

该脚本实现了直接使用有监督CNN进行SDB检测的功能，作为对比学习方法的基线。
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
import argparse
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sleepfm.config import Config


class SupervisedCNN(nn.Module):
    """用于SDB检测的有监督CNN模型"""
    
    def __init__(self, in_channels=3, num_classes=1):
        """
        初始化模型
        
        参数:
        in_channels: 输入通道数，默认为3（呼吸信号）
        num_classes: 输出类别数，默认为1（二分类问题）
        """
        super(SupervisedCNN, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 第四个卷积块
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 第五个卷积块
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            nn.Sigmoid()  # 二分类问题使用Sigmoid激活函数
        )
    
    def forward(self, x):
        """前向传播"""
        # 卷积特征提取
        features = self.conv_layers(x)
        
        # 全局池化
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)  # 展平
        
        # 分类
        output = self.classifier(pooled)
        
        return output
    
    
class MultiModalCNN(nn.Module):
    """多模态有监督CNN模型，支持呼吸信号、睡眠脑电和心电图数据"""
    
    def __init__(self, modality_channels=None, fusion='concat'):
        """
        初始化模型
        
        参数:
        modality_channels: 各模态的通道数字典，如{'Respiratory': 3, 'Sleep_Stages': 5, 'EKG': 1}
        fusion: 特征融合方法，可选'concat'、'attention'、'weighted'
        """
        super(MultiModalCNN, self).__init__()
        
        self.modality_channels = modality_channels or {
            'Respiratory': 3,  # CHEST, ABD, SaO2
            'Sleep_Stages': 5, # C3-M2, C4-M1, O1-M2, O2-M1, E1-M2
            'EKG': 1           # ECG
        }
        
        self.fusion = fusion
        self.modalities = list(self.modality_channels.keys())
        
        # 为每个模态创建独立的特征提取器
        self.feature_extractors = nn.ModuleDict()
        for modality, channels in self.modality_channels.items():
            self.feature_extractors[modality] = self._create_feature_extractor(channels)
        
        # 特征维度
        feature_dim = 256
        
        # 根据融合方法决定分类器输入维度
        if fusion == 'concat':
            # 拼接融合，输入维度是所有模态特征维度之和
            classifier_input_dim = feature_dim * len(self.modalities)
        else:
            # 加权或注意力融合，输入维度等于单个特征维度
            classifier_input_dim = feature_dim
            
            if fusion == 'attention':
                # 创建注意力层
                self.attention = nn.Sequential(
                    nn.Linear(feature_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
            elif fusion == 'weighted':
                # 创建可学习的权重参数
                self.modality_weights = nn.Parameter(
                    torch.ones(len(self.modalities)) / len(self.modalities)
                )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _create_feature_extractor(self, in_channels):
        """创建特征提取器"""
        return nn.Sequential(
            # 第一个卷积块
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 第二个卷积块
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 第三个卷积块
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 第四个卷积块
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 第五个卷积块
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # 全局平均池化
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self, x_dict):
        """
        前向传播
        
        参数:
        x_dict: 字典，包含各模态的输入数据，如{'Respiratory': x_resp, 'Sleep_Stages': x_sleep, 'EKG': x_ekg}
        """
        # 提取每个模态的特征
        features = {}
        for modality in self.modalities:
            if modality in x_dict:
                # 提取特征
                feat = self.feature_extractors[modality](x_dict[modality])
                # 展平
                features[modality] = feat.view(feat.size(0), -1)
        
        # 融合特征
        if self.fusion == 'concat':
            # 拼接融合
            combined_features = torch.cat([features[m] for m in self.modalities if m in features], dim=1)
        
        elif self.fusion == 'attention':
            # 注意力融合
            attention_scores = []
            for modality in self.modalities:
                if modality in features:
                    score = self.attention(features[modality])
                    attention_scores.append(score)
            
            # 计算注意力权重
            attention_weights = F.softmax(torch.cat(attention_scores, dim=1), dim=1)
            
            # 加权求和
            combined_features = torch.zeros_like(features[self.modalities[0]])
            for i, modality in enumerate(self.modalities):
                if modality in features:
                    combined_features += features[modality] * attention_weights[:, i:i+1]
        
        elif self.fusion == 'weighted':
            # 权重融合
            weights = F.softmax(self.modality_weights, dim=0)
            
            # 加权求和
            combined_features = torch.zeros_like(features[self.modalities[0]])
            for i, modality in enumerate(self.modalities):
                if modality in features:
                    combined_features += features[modality] * weights[i]
        
        # 分类
        output = self.classifier(combined_features)
        
        return output 

# 数据集类定义
class SDBDataset(Dataset):
    """SDB数据集类"""
    
    def __init__(self, X, y, modality_channels=None, transform=None):
        """
        初始化数据集
        
        参数:
        X: 特征数据
        y: 标签
        modality_channels: 模态通道字典，如{'Respiratory': [0,1,2], 'Sleep_Stages': [3,4,5,6,7], 'EKG': [8]}
        transform: 数据变换
        """
        self.X = X
        self.y = y
        self.transform = transform
        self.modality_channels = modality_channels
        self.use_multimodal = modality_channels is not None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 获取特征和标签
        features = self.X[idx]
        label = self.y[idx]
        
        # 应用变换
        if self.transform:
            features = self.transform(features)
        
        # 转换为PyTorch张量
        if self.use_multimodal:
            # 对于多模态数据，返回字典
            feature_dict = {}
            for modality, channels in self.modality_channels.items():
                modality_data = features[:, channels, :]  # 获取该模态的数据
                feature_dict[modality] = torch.FloatTensor(modality_data)
            
            return feature_dict, torch.FloatTensor([label])
        else:
            # 对于单一模态，直接返回张量
            return torch.FloatTensor(features), torch.FloatTensor([label])


def create_dataloaders(dataset, batch_size=32, num_workers=4):
    """
    创建训练、验证和测试数据加载器
    
    参数:
    dataset: 数据集路径或已加载的数据集
    batch_size: 批次大小
    num_workers: 工作线程数
    
    返回:
    dict: 包含训练、验证和测试数据加载器的字典
    """
    # 加载数据集
    if isinstance(dataset, str):
        with open(dataset, 'rb') as f:
            dataset = pickle.load(f)
    
    # 创建数据加载器
    dataloaders = {}
    for split in ['train', 'valid', 'test']:
        # 获取数据
        X = dataset[split]['X']
        y = dataset[split]['y']
        
        # 创建数据集
        if 'modality_channels' in dataset:
            # 多模态数据集
            data = SDBDataset(X, y, modality_channels=dataset['modality_channels'])
        else:
            # 单模态数据集
            data = SDBDataset(X, y)
        
        # 创建数据加载器
        shuffle = (split == 'train')  # 只对训练集打乱
        dataloaders[split] = DataLoader(
            data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
    
    return dataloaders 

def train_model(model, dataloaders, device, num_epochs=50, learning_rate=1e-4, weight_decay=1e-5):
    """
    训练模型
    
    参数:
    model: 模型
    dataloaders: 数据加载器字典
    device: 设备
    num_epochs: 训练轮数
    learning_rate: 学习率
    weight_decay: 权重衰减
    
    返回:
    tuple: (训练后的模型, 训练历史)
    """
    # 设置损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_auprc': []
    }
    
    # 用于早停的变量
    best_val_metric = 0
    patience = 5
    no_improve_epochs = 0
    best_model_weights = model.state_dict().copy()
    
    # 训练循环
    start_time = time.time()
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # 每个epoch有训练和验证阶段
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            y_true = []
            y_pred = []
            y_scores = []
            
            # 遍历数据
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                # 将数据移至设备
                if isinstance(inputs, dict):
                    # 多模态数据
                    for modality in inputs:
                        inputs[modality] = inputs[modality].to(device)
                else:
                    # 单模态数据
                    inputs = inputs.to(device)
                
                labels = labels.to(device)
                
                # 梯度归零
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播（仅训练阶段）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * labels.size(0)
                
                # 获取预测
                preds = (outputs > 0.5).float()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_scores.extend(outputs.detach().cpu().numpy())
            
            # 计算epoch统计
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()
            y_scores = np.array(y_scores).flatten()
            
            # 计算准确率
            epoch_acc = accuracy_score(y_true, y_pred)
            
            # 记录历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                logger.info(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
            else:
                # 计算验证集AUC和AUPRC
                try:
                    epoch_auc = roc_auc_score(y_true, y_scores)
                    precision, recall, _ = precision_recall_curve(y_true, y_scores)
                    epoch_auprc = auc(recall, precision)
                except:
                    epoch_auc = 0
                    epoch_auprc = 0
                
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                history['val_auc'].append(epoch_auc)
                history['val_auprc'].append(epoch_auprc)
                
                logger.info(f"Valid Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, AUC: {epoch_auc:.4f}, AUPRC: {epoch_auprc:.4f}")
                
                # 早停检查
                current_metric = (epoch_auc + epoch_auprc) / 2  # 使用AUC和AUPRC的平均值
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    no_improve_epochs = 0
                    best_model_weights = model.state_dict().copy()
                else:
                    no_improve_epochs += 1
        
        # 早停
        if no_improve_epochs >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_weights)
    
    # 记录训练时间
    time_elapsed = time.time() - start_time
    logger.info(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    logger.info(f"Best validation metric: {best_val_metric:.4f}")
    
    return model, history


def evaluate_model(model, dataloader, device):
    """
    评估模型
    
    参数:
    model: 模型
    dataloader: 数据加载器
    device: 设备
    
    返回:
    dict: 评估结果
    """
    model.eval()
    
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            # 将数据移至设备
            if isinstance(inputs, dict):
                # 多模态数据
                for modality in inputs:
                    inputs[modality] = inputs[modality].to(device)
            else:
                # 单模态数据
                inputs = inputs.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 获取预测
            preds = (outputs > 0.5).float()
            
            # 添加到列表
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())
    
    # 转换为numpy数组
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_scores = np.array(y_scores).flatten()
    
    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_scores)
    
    # 计算PR曲线下面积
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall_curve, precision_curve)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 结果
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'confusion_matrix': cm
    }
    
    # 打印结果
    logger.info(f"Test Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  AUROC: {auroc:.4f}")
    logger.info(f"  AUPRC: {auprc:.4f}")
    
    return results


def visualize_results(history, results, save_dir):
    """
    可视化结果
    
    参数:
    history: 训练历史
    results: 评估结果
    save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制AUC和AUPRC曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.plot(history['val_auprc'], label='Validation AUPRC')
    plt.title('AUC and AUPRC Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'auc_auprc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制混淆矩阵
    cm = results['confusion_matrix']
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Non-SDB', 'SDB'])
    plt.yticks([0, 1], ['Non-SDB', 'SDB'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 在矩阵中添加文本
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存结果到CSV
    result_df = pd.DataFrame([results])
    result_df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    
    logger.info(f"Results visualized and saved to {save_dir}") 

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练有监督CNN模型用于SDB检测")
    
    # 数据参数
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='数据集路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, choices=['cnn', 'multimodal'],
                       default='cnn', help='模型类型')
    parser.add_argument('--fusion', type=str, choices=['concat', 'attention', 'weighted'],
                       default='concat', help='多模态融合方法')
    parser.add_argument('--modalities', nargs='+', 
                       default=['Respiratory'], 
                       help='使用的模态')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID，-1表示使用CPU')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 选择设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # 设置输出目录
    config = Config()
    output_dir = args.output_dir or os.path.join(config.MODEL_PATH, 'supervised_cnn')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    logger.info(f"Loading dataset from: {args.dataset_path}")
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # 创建数据加载器
    dataloaders = create_dataloaders(dataset, args.batch_size)
    
    # 创建模型
    if args.model_type == 'cnn':
        # 单模态模型
        in_channels = 3  # 默认使用呼吸信号(CHEST, ABD, SaO2)
        model = SupervisedCNN(in_channels=in_channels, num_classes=1)
        logger.info(f"Created supervised CNN model with {in_channels} input channels")
    else:
        # 多模态模型
        modality_channels = {}
        for modality in args.modalities:
            if modality == 'Respiratory':
                modality_channels[modality] = 3
            elif modality == 'Sleep_Stages':
                modality_channels[modality] = 5
            elif modality == 'EKG':
                modality_channels[modality] = 1
        
        model = MultiModalCNN(modality_channels=modality_channels, fusion=args.fusion)
        logger.info(f"Created multimodal CNN model with modalities: {modality_channels}")
    
    # 将模型移至设备
    model = model.to(device)
    
    # 训练模型
    logger.info("Starting training...")
    model, history = train_model(
        model, 
        dataloaders, 
        device, 
        num_epochs=args.num_epochs, 
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # 评估模型
    logger.info("Evaluating model on test set...")
    results = evaluate_model(model, dataloaders['test'], device)
    
    # 可视化结果
    logger.info("Visualizing results...")
    visualize_results(history, results, output_dir)
    
    # 保存模型
    model_path = os.path.join(output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # 保存训练历史和结果
    history_path = os.path.join(output_dir, 'history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    results_path = os.path.join(output_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    # 配置日志
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    
    main() 