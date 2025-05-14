import torch
import cv2
import os
import numpy as np
import pandas
import pickle
import torchvision
import random
import math
import time
from config import (CHANNEL_DATA, ALL_CHANNELS, CHANNEL_DATA_IDS)
from loguru import logger
import shutil
import sys
sys.path.append("../")
from config import EVENT_TO_ID, LABELS_DICT, SDB_CONFIG


class EventDatasetSupervised(torchvision.datasets.VisionDataset):
    def __init__(self, root, split="train", modality_type="sleep_stages"):
        start = time.time()
        self.split = split
        self.modality_type = modality_type

        with open(root, "rb") as f:
            self.dataset = pickle.load(f)

        if split == "combined":
            self.dataset = self.dataset["pretrain"] + self.dataset["train"]
        else:
            self.dataset = self.dataset[split]
        
        logger.info(f"Loading dataset: {split} ({len(self.dataset)}), time: {time.time() - start}")
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        signal, event = self.dataset[index]
        
        return signal, event


class EventDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, split="train", modality_type=["respiratory", "sleep_stages", "ekg"]):
        start = time.time()
        self.split = split
        if isinstance(modality_type, list):
            self.modality_type = modality_type
        else:
            self.modality_type = [modality_type]

        with open(root, "rb") as f:
            self.dataset = pickle.load(f)

        self.dataset = self.dataset[split]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        data_path = self.dataset[index][0]
        data = np.load(data_path)
        
        target = []
        for t in self.modality_type:
            if t == "respiratory":
                resp_data = data[CHANNEL_DATA_IDS["Respiratory"]]
                target.append(resp_data)
            elif t == "sleep_stages":
                sleep_data = data[CHANNEL_DATA_IDS["Sleep_Stages"]]
                target.append(sleep_data)
            elif t == "ekg":
                ekg_data = data[CHANNEL_DATA_IDS["EKG"]]
                target.append(ekg_data)
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')
        
        return target


class SDBDataset(torch.utils.data.Dataset):
    """用于SDB（睡眠呼吸障碍）检测的数据集类
    
    该数据集类用于加载和处理SDB检测任务的数据。可以直接加载pickle格式的数据文件，
    也可以加载原始信号数据并提取特征。
    """
    
    def __init__(self, features, labels, transform=None, record_info=None):
        """
        初始化SDB数据集
        
        参数:
        features: numpy数组，特征数据
        labels: numpy数组，标签数据（0=非SDB, 1=SDB）
        transform: 数据转换函数（可选）
        record_info: 记录信息，包含每个样本的元数据（可选）
        """
        self.features = features
        self.labels = labels
        self.transform = transform
        self.record_info = record_info
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # 获取特征和标签
        features = self.features[idx]
        label = self.labels[idx]
        
        # 应用变换（如果有）
        if self.transform:
            features = self.transform(features)
        
        # 转换为PyTorch张量
        features = torch.FloatTensor(features)
        label = torch.FloatTensor([label])
        
        return features, label


class SDBEmbeddingDataset(torch.utils.data.Dataset):
    """用于使用预训练SleepFM模型生成的嵌入向量进行SDB检测的数据集类"""
    
    def __init__(self, embeddings, labels, record_info=None):
        """
        初始化SDB嵌入数据集
        
        参数:
        embeddings: numpy数组，嵌入向量
        labels: numpy数组，标签数据（0=非SDB, 1=SDB）
        record_info: 记录信息，包含每个样本的元数据（可选）
        """
        self.embeddings = embeddings
        self.labels = labels
        self.record_info = record_info
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        # 获取嵌入向量和标签
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        
        # 转换为PyTorch张量
        embedding = torch.FloatTensor(embedding)
        label = torch.FloatTensor([label])
        
        return embedding, label


_cache = {}
def cache_csv(path, sep=None):
    if path in _cache:
        return _cache[path]
    else:
        x = pandas.read_csv(path, sep=sep)
        _cache[path] = x
        return x

_cache = {}
def cache_pkl(path):
    if path in _cache:
        return _cache[path]
    else:
        with open(path, "rb") as f:
            x = pickle.load(f)
        _cache[path] = x
        return x


def load_sdb_dataset(dataset_path):
    """
    加载SDB数据集
    
    参数:
    dataset_path: 数据集路径
    
    返回:
    dict: 包含训练/验证/测试数据的字典
    """
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # 检查数据集格式
        required_keys = ['train', 'valid', 'test']
        for key in required_keys:
            if key not in dataset:
                logger.error(f"数据集缺少 '{key}' 部分")
                return None
            
            # 检查每个部分是否包含X和y
            if 'X' not in dataset[key] or 'y' not in dataset[key]:
                logger.error(f"数据集的 '{key}' 部分缺少特征或标签")
                return None
                
        return dataset
    
    except Exception as e:
        logger.error(f"加载数据集时出错: {str(e)}")
        return None


class MultiModalSDBDataset(torch.utils.data.Dataset):
    """多模态SDB数据集类，支持使用多种生理信号进行SDB检测"""
    
    def __init__(self, X, y, modality_channels=None, transform=None):
        """
        初始化
        
        参数:
        X: 特征数据，形状为(n_samples, n_channels, signal_length)
        y: 标签，形状为(n_samples,)
        modality_channels: 各模态对应的通道索引，如{'Respiratory': [0, 1, 2], 'EKG': [3], 'Sleep_Stages': [4, 5, 6, 7, 8]}
        transform: 数据增强/转换方法
        """
        self.X = X
        self.y = y
        self.transform = transform
        self.modality_channels = modality_channels or {}
        
        # 确保标签是浮点型
        if isinstance(self.y, np.ndarray):
            self.y = self.y.astype(np.float32)
            
        logger.info(f"创建多模态SDB数据集: {len(self.X)} 样本")
        logger.info(f"特征形状: {self.X.shape}")
        logger.info(f"模态通道: {self.modality_channels}")
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """获取特定样本"""
        # 获取输入特征
        x = self.X[idx]
        
        # 应用数据转换
        if self.transform:
            x = self.transform(x)
            
        # 确保输入是浮点型
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        
        # 获取标签
        label = self.y[idx]
        if isinstance(label, np.ndarray) or isinstance(label, list):
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor([label], dtype=torch.float32)
            
        return x, label
    
    def get_modality_data(self, modality):
        """获取特定模态的数据
        
        参数:
        modality: 模态名称，如'Respiratory', 'EKG', 'Sleep_Stages'
        
        返回:
        特定模态的数据子集
        """
        if modality not in self.modality_channels or not self.modality_channels[modality]:
            logger.warning(f"模态 {modality} 没有可用的通道")
            return None
        
        # 提取该模态的通道
        channels = self.modality_channels[modality]
        modality_data = self.X[:, channels, :]
        
        return modality_data
    
    def create_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
        """创建数据加载器
        
        参数:
        batch_size: 批次大小
        shuffle: 是否打乱顺序
        num_workers: 工作线程数
        
        返回:
        数据加载器
        """
        return torch.utils.data.DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )


class MultiModalSDBFeatureDataset(torch.utils.data.Dataset):
    """使用提取的特征进行SDB分类的数据集"""
    
    def __init__(self, X_dict, y, transform=None):
        """
        初始化
        
        参数:
        X_dict: 各模态特征字典，如{'Respiratory': X_resp, 'EKG': X_ekg, 'Sleep_Stages': X_eeg}
        y: 标签
        transform: 数据转换方法
        """
        self.X_dict = X_dict
        self.y = y
        self.transform = transform
        self.modalities = list(X_dict.keys())
        
        # 确保标签是浮点型
        if isinstance(self.y, np.ndarray):
            self.y = self.y.astype(np.float32)
            
        # 计算总样本数
        self.n_samples = len(self.y)
        
        logger.info(f"创建多模态特征数据集: {self.n_samples} 样本")
        logger.info(f"可用模态: {self.modalities}")
        for modality, features in self.X_dict.items():
            logger.info(f"  {modality} 特征形状: {features.shape}")
    
    def __len__(self):
        """返回数据集大小"""
        return self.n_samples
    
    def __getitem__(self, idx):
        """获取特定样本"""
        # 收集每个模态的特征
        features = {}
        for modality in self.modalities:
            features[modality] = self.X_dict[modality][idx]
            
            # 应用数据转换
            if self.transform:
                features[modality] = self.transform(features[modality])
                
            # 确保是浮点型
            if isinstance(features[modality], np.ndarray):
                features[modality] = torch.from_numpy(features[modality].astype(np.float32))
        
        # 获取标签
        label = self.y[idx]
        if isinstance(label, np.ndarray) or isinstance(label, list):
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor([label], dtype=torch.float32)
            
        return features, label
    
    def create_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
        """创建数据加载器"""
        return torch.utils.data.DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )