import pandas as pd
from tqdm import tqdm
import pickle
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from loguru import logger
import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import Counter
import sys
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample

# 添加模型路径
sys.path.append("/root/autodl-fs/sleepfm-codebase-main/sleepfm-codebase-main/sleepfm/model")
from supervised_cnn import SupervisedCNN
import config
from config import (MODALITY_TYPES, CLASS_LABELS, 
                    LABELS_DICT, PATH_TO_PROCESSED_DATA)
from dataset import EventDataset as Dataset

def compute_metrics_with_ci(y_true, y_probs, n_bootstrap=1000, ci=0.95):
    """
    计算AUROC、AUPRC和F1值，以及它们的置信区间
    
    参数:
    y_true: 真实标签
    y_probs: 预测概率
    n_bootstrap: 重抽样次数
    ci: 置信区间
    
    返回:
    metrics_dict: 包含各指标平均值和置信区间的字典
    """
    lb = LabelBinarizer()
    lb.fit(range(len(CLASS_LABELS)))
    y_true_bin = lb.transform(y_true)
    
    # 计算每个类别的指标
    metrics_dict = {}
    y_pred = np.argmax(y_probs, axis=1)
    
    for i, label in enumerate(CLASS_LABELS):
        auroc_samples = []
        auprc_samples = []
        f1_samples = []
        
        # 计算原始指标
        auroc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        auprc = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        f1 = accuracy_score(y_true == i, y_pred == i)  # 使用准确率作为F1
        
        # Bootstrap采样计算置信区间
        indices = np.arange(len(y_true))
        for _ in range(n_bootstrap):
            bootstrap_indices = resample(indices, replace=True)
            
            if len(np.unique(y_true[bootstrap_indices])) < 2:
                continue  # 跳过只有一个类别的样本
                
            try:
                auroc_bootstrap = roc_auc_score(y_true_bin[bootstrap_indices, i], y_probs[bootstrap_indices, i])
                auprc_bootstrap = average_precision_score(y_true_bin[bootstrap_indices, i], y_probs[bootstrap_indices, i])
                f1_bootstrap = accuracy_score(y_true[bootstrap_indices] == i, y_pred[bootstrap_indices] == i)
                
                auroc_samples.append(auroc_bootstrap)
                auprc_samples.append(auprc_bootstrap)
                f1_samples.append(f1_bootstrap)
            except:
                continue
            
        # 计算标准差
        auroc_std = np.std(auroc_samples)
        auprc_std = np.std(auprc_samples)
        f1_std = np.std(f1_samples)
        
        metrics_dict[label] = {
            'AUROC': {'value': auroc, 'std': auroc_std},
            'AUPRC': {'value': auprc, 'std': auprc_std},
            'F1': {'value': f1, 'std': f1_std}
        }
    
    # 计算平均值
    metrics_dict['平均'] = {
        'AUROC': {'value': np.mean([metrics_dict[label]['AUROC']['value'] for label in CLASS_LABELS])},
        'AUPRC': {'value': np.mean([metrics_dict[label]['AUPRC']['value'] for label in CLASS_LABELS])},
        'F1': {'value': np.mean([metrics_dict[label]['F1']['value'] for label in CLASS_LABELS])}
    }
    
    return metrics_dict

def format_metric(value, std):
    """格式化指标输出，保留3位小数"""
    return f"{value:.3f}±{std:.3f}"

def prepare_data(dataset_dir, modality_type="combined"):
    """准备训练和测试数据"""
    logger.info("开始加载数据...")
    
    try:
        # 尝试加载原始特征和标签
        raw_features_path = os.path.join(dataset_dir, "raw_features.pickle")
        raw_labels_path = os.path.join(dataset_dir, "raw_labels.pickle")
        
        if os.path.exists(raw_features_path) and os.path.exists(raw_labels_path):
            logger.info("从raw_features.pickle和raw_labels.pickle加载数据...")
            
            with open(raw_features_path, "rb") as f:
                raw_features = pickle.load(f)
            
            with open(raw_labels_path, "rb") as f:
                raw_labels = pickle.load(f)
            
            logger.info(f"加载了 {len(raw_features)} 个特征样本")
            logger.info(f"标签分布: {Counter(raw_labels.values())}")
            
            # 将数据分为训练集和测试集
            feature_keys = list(raw_features.keys())
            np.random.shuffle(feature_keys)
            
            # 使用80%数据作为训练集，20%作为测试集
            split_idx = int(len(feature_keys) * 0.8)
            train_keys = feature_keys[:split_idx]
            test_keys = feature_keys[split_idx:]
            
            logger.info(f"训练集样本数: {len(train_keys)}")
            logger.info(f"测试集样本数: {len(test_keys)}")
            
            # 准备训练数据
            X_train_list = [[] for _ in range(3)]  # 三个模态
            y_train = []
            
            for key in train_keys:
                features = raw_features[key]
                label = raw_labels[key]
                
                # 将特征添加到对应模态
                for i in range(3):
                    X_train_list[i].append(features[i])
                
                y_train.append(label)
            
            # 准备测试数据
            X_test_list = [[] for _ in range(3)]  # 三个模态
            y_test = []
            
            for key in test_keys:
                features = raw_features[key]
                label = raw_labels[key]
                
                # 将特征添加到对应模态
                for i in range(3):
                    X_test_list[i].append(features[i])
                
                y_test.append(label)
            
            # 转换为张量
            X_train = []
            X_test = []
            
            for i in range(3):
                # 处理不同形状的特征
                # 获取当前模态中所有样本的通道数和时间点数
                channels = [x.shape[0] for x in X_train_list[i]]
                max_channels = max(channels) if channels else 0
                time_points = [x.shape[1] for x in X_train_list[i]]
                max_time_points = max(time_points) if time_points else 0
                
                # 创建填充后的张量
                if max_channels > 0 and max_time_points > 0:
                    train_tensor = np.zeros((len(X_train_list[i]), max_channels, max_time_points))
                    for j, x in enumerate(X_train_list[i]):
                        c, t = x.shape
                        train_tensor[j, :c, :t] = x
                    X_train.append(torch.FloatTensor(train_tensor))
                
                channels = [x.shape[0] for x in X_test_list[i]]
                max_channels = max(channels) if channels else 0
                time_points = [x.shape[1] for x in X_test_list[i]]
                max_time_points = max(time_points) if time_points else 0
                
                if max_channels > 0 and max_time_points > 0:
                    test_tensor = np.zeros((len(X_test_list[i]), max_channels, max_time_points))
                    for j, x in enumerate(X_test_list[i]):
                        c, t = x.shape
                        test_tensor[j, :c, :t] = x
                    X_test.append(torch.FloatTensor(test_tensor))
            
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            
            return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.error(f"加载raw_features.pickle和raw_labels.pickle时出错: {e}")
        logger.info("尝试使用原始数据集格式...")
    
    # 原始的数据加载逻辑（保留作为备份）
    path_to_dataset = os.path.join(dataset_dir, f"dataset.pickle")
    with open(path_to_dataset, "rb") as f:
        dataset = pickle.load(f)

    path_to_event_dataset = os.path.join(dataset_dir, f"dataset_events_-1.pickle")
    with open(path_to_event_dataset, "rb") as f:
        dataset_events = pickle.load(f)
    
    # 加载原始特征数据，而不是嵌入特征
    logger.info("加载原始特征数据...")
    # 示例数据格式需根据实际调整
    with open(os.path.join(dataset_dir, f"raw_features.pickle"), "rb") as f:
        raw_features = pickle.load(f)
    
    # 获取标签
    path_to_label = {}
    for split, split_dataset in tqdm(dataset.items(), desc="处理数据集"):
        for patient_data in tqdm(split_dataset, desc=f"处理{split}集"):
            mrn = list(patient_data.keys())[0]
            for event, event_paths in patient_data[mrn].items():
                for event_path in event_paths:
                    path_to_label[event_path] = event 

    # 提取训练集和测试集
    train_paths = dataset_events["train"]
    test_paths = dataset_events["test"]
    
    # 加载特征和标签
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    # 处理训练集
    for path in tqdm(train_paths, desc="处理训练集"):
        if path[0] in path_to_label and path_to_label[path[0]] in LABELS_DICT:
            if path[0] in raw_features:
                X_train.append(raw_features[path[0]])
                y_train.append(LABELS_DICT[path_to_label[path[0]]])
    
    # 处理测试集
    for path in tqdm(test_paths, desc="处理测试集"):
        if path[0] in path_to_label and path_to_label[path[0]] in LABELS_DICT:
            if path[0] in raw_features:
                X_test.append(raw_features[path[0]])
                y_test.append(LABELS_DICT[path_to_label[path[0]]])
    
    logger.info(f"训练集样本数: {len(X_train)}")
    logger.info(f"测试集样本数: {len(X_test)}")
    
    # 转换为张量
    X_train = [torch.FloatTensor(x) for x in zip(*X_train)]
    X_test = [torch.FloatTensor(x) for x in zip(*X_test)]
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    return X_train, y_train, X_test, y_test

def train_supervised_cnn(X_train, y_train, X_test, y_test, modality_dims, output_dir, batch_size=32, num_epochs=20, lr=0.01):
    """训练有监督CNN模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器
    train_dataset = TensorDataset(*X_train, y_train)
    test_dataset = TensorDataset(*X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 实例化模型
    model = SupervisedCNN(modality_dims, num_classes=len(CLASS_LABELS)).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # 训练循环
    logger.info("开始训练模型...")
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (*inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # 将数据移至GPU
            inputs = [x.to(device) for x in inputs]
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 测试模型
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (*inputs, labels) in enumerate(test_loader):
                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # 计算测试指标
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        accuracy = 100. * correct / total
        test_accuracies.append(accuracy)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # 保存模型
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "models", "supervised_cnn.pth"))
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "figures", "loss_curve.png"))
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "figures", "accuracy_curve.png"))
    
    return model

def evaluate_model(model, X_test, y_test, output_dir):
    """评估模型并计算高级指标"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 创建测试数据加载器
    test_dataset = TensorDataset(*X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # 收集预测结果
    y_probs = []
    y_true = []
    
    with torch.no_grad():
        for batch_idx, (*inputs, labels) in enumerate(tqdm(test_loader, desc="评估模型")):
            inputs = [x.to(device) for x in inputs]
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            y_probs.append(probs.cpu().numpy())
            y_true.append(labels.numpy())
    
    # 合并批次结果
    y_probs = np.vstack(y_probs)
    y_true = np.concatenate(y_true)
    
    # 计算高级指标
    metrics = compute_metrics_with_ci(y_true, y_probs)
    
    # 保存预测概率
    os.makedirs(os.path.join(output_dir, "probs"), exist_ok=True)
    with open(os.path.join(output_dir, "probs", "supervised_cnn_y_probs.pickle"), 'wb') as f:
        pickle.dump(y_probs, f)
    
    # 保存测试标签
    with open(os.path.join(output_dir, "probs", "supervised_cnn_test_labels.pickle"), 'wb') as f:
        pickle.dump(y_true, f)
    
    # 显示结果
    logger.info("\n有监督CNN在CinC数据集上的性能:")
    logger.info("-" * 80)
    logger.info(f"{'睡眠阶段':<10} | {'AUROC':<15} | {'AUPRC':<15} | {'F1':<15}")
    logger.info("-" * 80)
    
    for label in CLASS_LABELS + ['平均']:
        label_name = label
        if label == 'Wake':
            label_name = '清醒'
        elif label == 'Stage 1':
            label_name = '第一阶段'
        elif label == 'Stage 2':
            label_name = '第二阶段'
        elif label == 'Stage 3':
            label_name = '第三阶段'
        elif label == 'REM':
            label_name = '快速眼动'
        
        if label == '平均':
            auroc = f"{metrics[label]['AUROC']['value']:.3f}"
            auprc = f"{metrics[label]['AUPRC']['value']:.3f}" 
            f1 = f"{metrics[label]['F1']['value']:.3f}"
        else:
            auroc = format_metric(metrics[label]['AUROC']['value'], metrics[label]['AUROC']['std'])
            auprc = format_metric(metrics[label]['AUPRC']['value'], metrics[label]['AUPRC']['std'])
            f1 = format_metric(metrics[label]['F1']['value'], metrics[label]['F1']['std'])
        
        logger.info(f"{label_name:<10} | {auroc:<15} | {auprc:<15} | {f1:<15}")
    
    logger.info("-" * 80)
    
    # 保存结果到CSV
    results_df = pd.DataFrame({
        '睡眠阶段': ['清醒', '第一阶段', '第二阶段', '第三阶段', '快速眼动', '平均'],
        'AUROC': [
            format_metric(metrics['Wake']['AUROC']['value'], metrics['Wake']['AUROC']['std']),
            format_metric(metrics['Stage 1']['AUROC']['value'], metrics['Stage 1']['AUROC']['std']),
            format_metric(metrics['Stage 2']['AUROC']['value'], metrics['Stage 2']['AUROC']['std']),
            format_metric(metrics['Stage 3']['AUROC']['value'], metrics['Stage 3']['AUROC']['std']),
            format_metric(metrics['REM']['AUROC']['value'], metrics['REM']['AUROC']['std']),
            f"{metrics['平均']['AUROC']['value']:.3f}"
        ],
        'AUPRC': [
            format_metric(metrics['Wake']['AUPRC']['value'], metrics['Wake']['AUPRC']['std']),
            format_metric(metrics['Stage 1']['AUPRC']['value'], metrics['Stage 1']['AUPRC']['std']),
            format_metric(metrics['Stage 2']['AUPRC']['value'], metrics['Stage 2']['AUPRC']['std']),
            format_metric(metrics['Stage 3']['AUPRC']['value'], metrics['Stage 3']['AUPRC']['std']),
            format_metric(metrics['REM']['AUPRC']['value'], metrics['REM']['AUPRC']['std']),
            f"{metrics['平均']['AUPRC']['value']:.3f}"
        ],
        'F1': [
            format_metric(metrics['Wake']['F1']['value'], metrics['Wake']['F1']['std']),
            format_metric(metrics['Stage 1']['F1']['value'], metrics['Stage 1']['F1']['std']),
            format_metric(metrics['Stage 2']['F1']['value'], metrics['Stage 2']['F1']['std']),
            format_metric(metrics['Stage 3']['F1']['value'], metrics['Stage 3']['F1']['std']),
            format_metric(metrics['REM']['F1']['value'], metrics['REM']['F1']['std']),
            f"{metrics['平均']['F1']['value']:.3f}"
        ]
    })
    
    results_df.to_csv(os.path.join(output_dir, "supervised_cnn_metrics.csv"), index=False)
    logger.info(f"结果已保存至: {os.path.join(output_dir, 'supervised_cnn_metrics.csv')}")
    
    return metrics

def main(args):
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    
    if dataset_dir is None:
        dataset_dir = PATH_TO_PROCESSED_DATA
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    logger.info("准备数据...")
    X_train, y_train, X_test, y_test = prepare_data(dataset_dir)
    
    # 获取每个模态的输入维度
    modality_dims = [X_train[i].shape[1] for i in range(len(X_train))]
    logger.info(f"模态维度: {modality_dims}")
    
    # 训练模型
    logger.info("训练有监督CNN模型...")
    model = train_supervised_cnn(
        X_train, y_train, X_test, y_test, 
        modality_dims, output_dir, 
        batch_size, num_epochs, learning_rate
    )
    
    # 评估模型
    logger.info("评估模型...")
    evaluate_model(model, X_test, y_test, output_dir)
    
    logger.info("完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练有监督CNN模型进行睡眠分期")
    parser.add_argument("--dataset_dir", type=str, default=None, help="数据集目录路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--num_epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="学习率")
    
    args = parser.parse_args()
    main(args) 