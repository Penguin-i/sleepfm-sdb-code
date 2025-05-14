import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    """1D卷积块，类似于EfficientNet中的块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.swish = nn.SiLU()  # Swish激活函数 (SiLU)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.swish(x)
        return x

class SqueezeExcitation(nn.Module):
    """挤压激励模块"""
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    """移动瓶颈卷积块 (MBConv)"""
    def __init__(self, in_channels, out_channels, expand_ratio=4, kernel_size=3, stride=1, reduction=4, dropout_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.skip_connection = in_channels == out_channels and stride == 1
        
        expanded_channels = in_channels * expand_ratio
        
        layers = []
        # 扩展
        if expanded_channels != in_channels:
            layers.append(ConvBlock(in_channels, expanded_channels, kernel_size=1))
        
        # 深度可分离卷积
        layers.extend([
            # 使用组卷积确保通道数匹配
            ConvBlock(expanded_channels, expanded_channels, kernel_size, stride, padding=kernel_size//2, groups=expanded_channels),
            SqueezeExcitation(expanded_channels, reduction),
            # 投影卷积将通道数降到输出维度
            nn.Conv1d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ])
        
        self.layers = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        res = self.layers(x)
        if self.skip_connection:
            if self.training and self.dropout_rate > 0:
                res = F.dropout(res, p=self.dropout_rate, training=self.training)
            x = x + res
        return x

class SupervisedCNN(nn.Module):
    """基于简化CNN架构的有监督模型"""
    def __init__(self, modality_dims, num_classes=5, embed_dim=128, dropout_rate=0.3):
        """
        初始化有监督CNN模型
        
        参数:
        - modality_dims: list，每个模态的输入维度 [respiratory_dim, sleep_stages_dim, ekg_dim]
        - num_classes: int，分类任务的类别数量
        - embed_dim: int，嵌入空间维度
        - dropout_rate: float，dropout比率
        """
        super(SupervisedCNN, self).__init__()
        
        # 为每个模态构建编码器
        self.encoders = nn.ModuleList()
        
        # 通道配置更简单，避免太多的扩展
        config = [
            [32, 64, 128],  # 呼吸信号
            [32, 64, 128],  # 睡眠脑电
            [32, 64, 128]   # 心电图
        ]
        
        # 为每个模态构建编码器
        for i, input_dim in enumerate(modality_dims):
            # 构建一个简单的卷积网络
            layers = [
                # 第一层：输入维度 -> 32通道
                nn.Conv1d(input_dim, config[i][0], kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(config[i][0]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                
                # 第二层：32通道 -> 64通道
                nn.Conv1d(config[i][0], config[i][1], kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(config[i][1]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                
                # 第三层：64通道 -> 128通道
                nn.Conv1d(config[i][1], config[i][2], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(config[i][2]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            ]
            
            self.encoders.append(nn.Sequential(*layers))
        
        # 计算编码后的总特征维度
        total_features = sum([cfg[-1] for cfg in config])
        
        # 添加最终分类层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(total_features, embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: list，包含三个模态的输入张量 [respiratory, sleep_stages, ekg]
        
        返回:
        - logits: tensor，分类预测结果
        """
        features = []
        
        # 对每个模态进行编码
        for i, encoder in enumerate(self.encoders):
            features.append(encoder(x[i]))
        
        # 连接所有模态特征
        combined = torch.cat(features, dim=1)
        
        # 分类
        logits = self.classifier(combined)
        
        return logits 