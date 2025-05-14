import pandas as pd
import json
import sys
from collections import Counter
from pathlib import Path
import os
import csv
import mne
import numpy as np
from tqdm import tqdm
from loguru import logger
import pickle
import matplotlib.pyplot as plt
from typing import Any, Union
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import torch


def save_data(data: Any, filename: str) -> None:
    """
    Save data to a file in either pickle or JSON format based on the file extension.

    Parameters:
    - data: The data to save.
    - filename: The name of the file to save the data to. Should have .pickle, .pkl, or .json extension.
    """
    if filename.endswith(('.pkl', '.pickle')):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    elif filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        raise ValueError("Filename must end with .pkl, .pickle, or .json")


def load_data(filename: str) -> Any:
    """
    Load data from a file in either pickle or JSON format based on the file extension.

    Parameters:
    - filename: The name of the file to load the data from. Should have .pickle, .pkl, or .json extension.

    Returns:
    - The loaded data.
    """
    if filename.endswith(('.pkl', '.pickle')):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Filename must end with .pkl, .pickle, or .json")



# Wrapper for getEDFFiles
def getEDFFilenames(path2check):
    edfFiles = getEDFFiles(path2check)
    return [str(i) for i in edfFiles]


def getEDFFiles(path2check):
    p = Path(path2check)
    # verify that we have an accurate directory
    # if so then list all .edf/.EDF files
    if p.is_dir():
        print("Checking", path2check, "for edf files.")
        edfFiles = list(p.glob("**/*.[EeRr][DdEe][FfCc]"))  # make search case-insensitive
        print('Removing any MSLT studies.')
        edfFiles = [edf for edf in edfFiles if not 'mslt' in edf.stem.lower()]
    else:
        print(path2check, " is not a valid directory.")
        edfFiles = []
    return edfFiles


def getSignalHeaders(edfFilename):
    try:
        # print("Reading headers from ", edfFilename)
        try:
            edfR = EdfReader(str(edfFilename))
            return edfR.getSignalHeaders()
        except:
            edfR = mne.io.read_raw_edf(str(edfFilename), verbose=False)
            return edfR.ch_names
    except:
        print("Could not read headers from {}".format(edfFilename))
        return []


def getChannelLabels(edfFilename):
    channelHeaders = getSignalHeaders(edfFilename)
    try:
        return [fields["label"] for fields in channelHeaders]
    except:
        return channelHeaders


def displaySetSelection(label_set):
    numCols = 4
    curItem = 0
    width = 30
    rowStr = ""
    for label, count in sorted(label_set.items()):
        rowStr += (f"{curItem}.".ljust(4) + f"{count}".rjust(4).ljust(5) + f"{label}").ljust(width)
        # rowStr = rowStr + str(str(str(curItem) + ".").ljust(4) + f"{count}".rjust(5) + f"{label}").ljust(
        #     width
        # )
        curItem = curItem + 1
        if curItem % numCols == 0:
            print(rowStr)
            rowStr = ""
    if len(rowStr) > 0:
        print(rowStr)


def getAllChannelLabels(path2check):
    edfFiles = getEDFFilenames(path2check)
    num_edfs = len(edfFiles)
    if num_edfs == 0:
        label_list = []
    else:
        label_set = getLabelSet(edfFiles)
        label_list = sorted(label_set)
    return label_set, num_edfs


def getAllChannelLabelsWithCounts(edfFiles):
    num_edfs = len(edfFiles)
    if num_edfs == 0:
        label_list = []
    else:
        label_list = []
        for edfFile in tqdm(edfFiles):
            [label_list.append(l) for l in getChannelLabels(edfFile)]
        label_set_counts = Counter(label_list)
    return label_set_counts, num_edfs


def getLabelSet(edfFiles):
    label_set = set()
    for edfFile in edfFiles:
        # only add unique channel labels to our set`
        label_set = label_set.union(set(getChannelLabels(edfFile)))
    return label_set


def read_events_file_as_df(file_name):
    df = pd.read_csv(file_name, sep='\t', header=None, encoding_errors='ignore')
    events = df[0].values
    header = events[1].split(',')
    rows = []
    for line in events[2:]:
        rows.append(next(csv.reader([line])))
    df = pd.DataFrame(rows, columns=header)

    return df, events[0]


def get_split(file_names, test_size=0.20, random_state=1):
    train_files, test_files = train_test_split(file_names, test_size=0.20, random_state=1)
    split = []

    for file_name in edf_filenames_pruned:
        if file_name in train_files:
            split.append("train")
        elif file_name in test_files:
            split.append("valid")

    return split, train_files, test_files


def get_all_edf_and_events_file_pair(path_to_dir: str):

    temp_dict = {}
    data_dict = {}

    for file_name in os.listdir(path_to_dir):

        file_prefix = file_name.split(".")[0]
        file_suffix = file_name.split(".")[-1]
        if file_suffix.upper() not in {"EDF", "EVTS", "EVTS_HUNEO"}:
            continue

        if file_prefix not in temp_dict:
            temp_dict[file_prefix] = {}

        if file_suffix.upper() == "EDF":
            temp_dict[file_prefix]["edf"] = os.path.join(path_to_dir, file_name)
        else:
            temp_dict[file_prefix]["evts"] = os.path.join(path_to_dir, file_name)
        if len(temp_dict[file_prefix]) == 2:
            data_dict[file_prefix] = temp_dict[file_prefix]

    return data_dict


def filter_edf_events_file_pair(data_dict, ALL_CHANNELS, num_of_files):

    edf_events_files_pruned = []

    for id, values in tqdm(data_dict.items()):
        edf_filename = values["edf"]
        event_filename = values["evts"]

        edf_raw = mne.io.read_raw_edf(edf_filename, verbose=False)
        channel_names = set(edf_raw.ch_names)
        if len(set(ALL_CHANNELS) - set(channel_names)) == 0:
            edf_events_files_pruned.append((edf_filename, event_filename))

        if num_of_files != -1 and len(edf_events_files_pruned) == num_of_files:
            break

    # assert len(edf_filenames_pruned) == len(event_filenames_pruned)

    return edf_events_files_pruned


def train_model(X_train, X_test, y_train, y_test, path_to_save, class_labels, model_name="logistic", n_bootstrap=100, alpha=0.95, max_iter=100, modality_type="combined"):
    """
    Train a classification model and generate evaluation metrics and plots.

    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    path_to_save : str
        Base path to save outputs
    class_labels : list
        List of class labels
    model_name : str, optional
        Name of the model to use ("logistic", "rf", or "xgb"), default="logistic"
    n_bootstrap : int, optional
        Number of bootstrap samples, default=100
    alpha : float, optional
        Confidence level for bootstrap intervals, default=0.95
    max_iter : int, optional
        Maximum number of iterations for logistic regression, default=100
    modality_type : str, optional
        Type of modality being used, default="combined"

    Returns:
    --------
    model : sklearn estimator
        Trained model
    y_probs : array-like
        Predicted probabilities
    class_report : dict
        Classification report with metrics
    """
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    logger.info(f"Starting training model {model_name}")

    # 配置分类器
    if model_name == "logistic":
        model = LogisticRegression(
            penalty='l2',
            max_iter=10000,
            class_weight='balanced',
            solver='lbfgs'
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,           # 限制树的最大深度
            min_samples_split=10,   # 增加分裂所需的最小样本数
            min_samples_leaf=4,     # 增加叶节点所需的最小样本数
            max_features='sqrt',    # 使用特征的平方根数量
            class_weight='balanced',
            n_jobs=-1,              # 使用所有可用的CPU核心
            random_state=42
        )
    elif model_name == "xgb":
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,          # 使用80%的样本训练每棵树
            colsample_bytree=0.8,   # 使用80%的特征训练每棵树
            tree_method='hist',     # 使用直方图算法加速训练
            n_jobs=-1,              # 使用所有可用的CPU核心
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}. Please choose from 'logistic', 'rf', or 'xgb'.")

    logger.info(f"Training {model_name} classifier...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.2f}")

    class_report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)

    # Create figures directory
    figures_dir = os.path.join(path_to_save, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'confusion_matrix_{model_name}_{modality_type}.png'))
    plt.close()

    # Save ROC curves
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test == i, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'roc_curves_{model_name}_{modality_type}.png'))
    plt.close()

    # Save PR curves
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(class_labels):
        precision, recall, _ = precision_recall_curve(y_test == i, y_probs[:, i])
        ap = average_precision_score(y_test == i, y_probs[:, i])
        plt.plot(recall, precision, label=f'{label} (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'pr_curves_{model_name}_{modality_type}.png'))
    plt.close()

    return model, y_probs, class_report


# SDB相关工具函数

def extract_sdb_events_from_annotations(annotations, sdb_keywords):
    """
    从注释中提取SDB事件

    参数:
    annotations: 包含注释的对象
    sdb_keywords: 用于识别SDB事件的关键词字典

    返回:
    list: SDB事件列表
    """
    sdb_events = []
    event_start = None
    event_type = None

    apnea_keywords = sdb_keywords.get('apnea', ['apnea', 'apnoea'])
    hypopnea_keywords = sdb_keywords.get('hypopnea', ['hypopnea', 'hypopoea'])

    for i in range(len(annotations.sample)):
        sample = annotations.sample[i]
        aux_note = annotations.aux_note[i] if hasattr(annotations, 'aux_note') and annotations.aux_note else ""

        # 检查是否是事件开始标记
        is_start = False
        if aux_note and aux_note.startswith('('):
            for keyword in apnea_keywords:
                if keyword in aux_note.lower():
                    event_type = "apnea"
                    is_start = True
                    break

            if not is_start:
                for keyword in hypopnea_keywords:
                    if keyword in aux_note.lower():
                        event_type = "hypopnea"
                        is_start = True
                        break

            if is_start:
                event_start = sample

        # 检查是否是事件结束标记
        is_end = False
        if event_start is not None and aux_note and aux_note.endswith(')'):
            for keyword in apnea_keywords + hypopnea_keywords:
                if keyword in aux_note.lower():
                    is_end = True
                    break

            if is_end:
                # 记录完整的事件
                sdb_events.append({
                    'start_sample': event_start,
                    'end_sample': sample,
                    'duration_samples': sample - event_start,
                    'event_type': event_type
                })
                # 重置事件标记
                event_start = None
                event_type = None

    return sdb_events


def classify_sdb_severity(ahi, severity_thresholds):
    """
    根据AHI（呼吸暂停低通气指数）对SDB严重程度进行分类

    参数:
    ahi: 呼吸暂停低通气指数
    severity_thresholds: 严重程度阈值字典

    返回:
    str: SDB严重程度分类
    """
    for severity, (min_val, max_val) in severity_thresholds.items():
        if min_val <= ahi < max_val:
            return severity

    return "unknown"


def create_window_labels(sdb_events, fs, n_samples, window_size_sec=30):
    """
    创建基于窗口的SDB标签

    参数:
    sdb_events: SDB事件列表
    fs: 采样率
    n_samples: 信号总长度（样本数）
    window_size_sec: 窗口大小（秒）

    返回:
    tuple: (窗口标签, 窗口起始位置)
    """
    window_size_samples = window_size_sec * fs
    n_windows = int(n_samples / window_size_samples)

    # 创建窗口标签
    window_labels = np.zeros(n_windows)
    window_positions = np.arange(n_windows) * window_size_samples

    # 对每个SDB事件，标记覆盖的窗口
    for event in sdb_events:
        start_sample = event['start_sample']
        end_sample = event['end_sample']

        # 计算该事件所在的窗口范围
        start_window = max(0, int(start_sample / window_size_samples))
        end_window = min(n_windows - 1, int(end_sample / window_size_samples))

        # 计算每个窗口中的事件占比
        for window_idx in range(start_window, end_window + 1):
            window_start = window_idx * window_size_samples
            window_end = window_start + window_size_samples

            # 计算事件在窗口内的重叠部分
            overlap_start = max(start_sample, window_start)
            overlap_end = min(end_sample, window_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            # 更新窗口标签（使用重叠占比）
            overlap_ratio = overlap_duration / window_size_samples
            window_labels[window_idx] += overlap_ratio

    # 确保标签在0到1之间
    window_labels = np.clip(window_labels, 0, 1)

    return window_labels, window_positions


def generate_binary_labels(window_labels, threshold=0.3):
    """
    根据窗口标签生成二分类标签

    参数:
    window_labels: 窗口标签（SDB事件占比）
    threshold: 阈值，窗口内SDB事件占比超过此值则标记为阳性

    返回:
    np.ndarray: 二分类标签（0=非SDB, 1=SDB）
    """
    return (window_labels >= threshold).astype(int)


def balance_dataset(X, y):
    """
    平衡数据集，使正负样本数量相等

    参数:
    X: 特征
    y: 标签

    返回:
    tuple: (平衡后的特征, 平衡后的标签)
    """
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    # 找出少数类样本数
    min_count = min(len(pos_indices), len(neg_indices))

    # 随机选择样本
    if len(pos_indices) > min_count:
        selected_pos = np.random.choice(pos_indices, min_count, replace=False)
        selected_neg = neg_indices
    else:
        selected_pos = pos_indices
        selected_neg = np.random.choice(neg_indices, min_count, replace=False)

    # 合并选中的样本索引
    selected_indices = np.concatenate([selected_pos, selected_neg])
    np.random.shuffle(selected_indices)

    # 返回平衡后的数据集
    return X[selected_indices], y[selected_indices]


def evaluate_sdb_model(model, X_test, y_test, output_dir, class_names=["Non-SDB", "SDB"], is_binary=True):
    """
    评估SDB检测模型

    参数:
    model: 训练好的模型
    X_test: 测试特征
    y_test: 测试标签
    output_dir: 输出目录
    class_names: 类别名称
    is_binary: 是否为二分类问题

    返回:
    dict: 评估指标
    """
    os.makedirs(output_dir, exist_ok=True)

    # 预测
    if hasattr(model, 'predict_proba'):
        y_probs = model.predict_proba(X_test)
        if is_binary:
            y_scores = y_probs[:, 1]
        else:
            y_scores = y_probs
    else:
        y_scores = model.predict(X_test)
        y_probs = np.column_stack((1 - y_scores, y_scores)) if is_binary else None

    y_pred = model.predict(X_test)

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)

    if is_binary:
        # 二分类评估
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_scores)

        # 计算PR曲线和AUPRC
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_scores)
        auprc = auc(recall_curve, precision_curve)

        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auroc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 绘制PR曲线
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, label=f'PR curve (AUC = {auprc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 保存指标
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auroc': auroc,
            'auprc': auprc
        }
    else:
        # 多分类评估
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        metrics = {
            'accuracy': accuracy,
            'report': report
        }

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 保存指标为CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    return metrics


def extract_modality_features(model, data_loader, device, modality='Respiratory'):
    """
    从预训练模型中提取特定模态的特征

    参数:
    model: 预训练模型
    data_loader: 数据加载器
    device: 计算设备
    modality: 模态名称

    返回:
    numpy.ndarray: 提取的特征
    numpy.ndarray: 对应的标签
    """
    logger.info(f"从{modality}模态提取特征...")
    model.eval()  # 设置为评估模式

    all_features = []
    all_labels = []

    # 禁用梯度计算以加速推理
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f"提取{modality}特征"):
            inputs = inputs.to(device)

            # 前向传播获取特征
            features = model(inputs)

            # 收集特征和标签
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 合并所有批次的结果
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)

    logger.info(f"提取了 {len(all_features)} 个 {modality} 特征向量")
    return all_features, all_labels


def fuse_multimodal_features(feature_dict, fusion_method='concat'):
    """
    融合多模态特征

    参数:
    feature_dict: 各模态特征的字典，如{'Respiratory': X_resp, 'EKG': X_ekg, 'Sleep_Stages': X_eeg}
    fusion_method: 融合方法，可选['concat', 'mean', 'max', 'weighted']

    返回:
    numpy.ndarray: 融合后的特征
    """
    if len(feature_dict) == 0:
        logger.error("没有特征可融合")
        return None

    if len(feature_dict) == 1:
        modality = list(feature_dict.keys())[0]
        logger.info(f"只有一个模态({modality})，不需要融合")
        return feature_dict[modality]

    logger.info(f"使用{fusion_method}方法融合{len(feature_dict)}个模态特征")

    if fusion_method == 'concat':
        # 连接所有特征
        return np.hstack([feature_dict[modality] for modality in feature_dict])

    elif fusion_method == 'mean':
        # 计算平均特征
        return np.mean([feature_dict[modality] for modality in feature_dict], axis=0)

    elif fusion_method == 'max':
        # 取最大值
        return np.max([feature_dict[modality] for modality in feature_dict], axis=0)

    elif fusion_method == 'weighted':
        # 加权融合(权重可调整)
        weights = {
            'Respiratory': 0.5,  # 呼吸信号权重
            'Sleep_Stages': 0.3, # 睡眠脑电权重
            'EKG': 0.2           # 心电图权重
        }

        weighted_features = []
        for modality, features in feature_dict.items():
            if modality in weights:
                weight = weights[modality]
                weighted_features.append(features * weight)

        return np.sum(weighted_features, axis=0)

    else:
        logger.error(f"未知的融合方法: {fusion_method}")
        return None


def compute_sdb_metrics(y_true, y_pred, y_score=None):
    """
    计算SDB检测的各种评估指标

    参数:
    y_true: 真实标签
    y_pred: 预测标签
    y_score: 预测概率分数(用于ROC和PR曲线)

    返回:
    dict: 包含各种评估指标的字典
    """
    # 确保标签是一维数组
    if len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.ravel()

    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 计算详细分类报告
    report = classification_report(y_true, y_pred,
                                  target_names=["非SDB", "SDB"],
                                  output_dict=True)

    # 计算其他常用指标
    acc = accuracy_score(y_true, y_pred)

    # 提取敏感性(召回率)和特异性
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'accuracy': acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': report['SDB']['f1-score'],
        'precision': report['SDB']['precision'],
        'recall': report['SDB']['recall'],
        'confusion_matrix': cm,
        'report': report
    }

    # 如果提供了概率分数，计算AUC和AUPRC
    if y_score is not None:
        if len(y_score.shape) > 1 and y_score.shape[1] == 1:
            y_score = y_score.ravel()

        # ROC曲线和AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # PR曲线和AUPRC
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

        metrics.update({
            'auc': roc_auc,
            'auprc': auprc,
            'roc_curve': {'fpr': fpr, 'tpr': tpr},
            'pr_curve': {'precision': precision, 'recall': recall}
        })

    return metrics


def plot_multimodal_performance(results_dict, save_path=None):
    """
    可视化多模态SDB检测性能比较

    参数:
    results_dict: 各模态/融合方法的结果字典，如
                  {'Respiratory': metrics1, 'EKG': metrics2, 'Sleep_Stages': metrics3, 'Fusion': metrics4}
    save_path: 保存图表的路径
    """
    # 提取各指标
    methods = list(results_dict.keys())
    metrics = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc', 'auprc']

    # 创建比较图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = []
        for method in methods:
            if metric in results_dict[method]:
                values.append(results_dict[method][metric])
            else:
                values.append(0)

        ax = axes[i]
        bars = ax.bar(methods, values)
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)

        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"性能比较图已保存到 {save_path}")

    plt.close()


def analyze_sdb_predictions(y_true, y_pred, y_score, record_info, threshold=0.5):
    """
    分析SDB预测结果，找出预测错误的样本和边界情况

    参数:
    y_true: 真实标签
    y_pred: 预测标签
    y_score: 预测概率分数
    record_info: 记录样本来源信息的列表
    threshold: 分类阈值

    返回:
    dict: 分析结果
    """
    # 确保所有输入都是一维数组
    if len(y_true.shape) > 1:
        y_true = y_true.ravel()

    if len(y_score.shape) > 1:
        y_score = y_score.ravel()

    # 获取不同类型的样本索引
    correct_indices = np.where(y_true == y_pred)[0]
    error_indices = np.where(y_true != y_pred)[0]

    # 根据预测概率找出边界情况
    boundary_indices = np.where(np.abs(y_score - threshold) < 0.1)[0]

    # 假阳性和假阴性
    false_positive = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negative = np.where((y_true == 1) & (y_pred == 0))[0]

    # 收集每个记录的预测性能
    record_metrics = {}

    for i, info in enumerate(record_info):
        record = info['record_name']

        if record not in record_metrics:
            record_metrics[record] = {
                'total': 0,
                'correct': 0,
                'error': 0,
                'false_positive': 0,
                'false_negative': 0,
                'sdb_ratio': info.get('sdb_ratio', 0)
            }

        record_metrics[record]['total'] += 1

        if i in correct_indices:
            record_metrics[record]['correct'] += 1

        if i in error_indices:
            record_metrics[record]['error'] += 1

        if i in false_positive:
            record_metrics[record]['false_positive'] += 1

        if i in false_negative:
            record_metrics[record]['false_negative'] += 1

    # 计算每个记录的准确率
    for record in record_metrics:
        metrics = record_metrics[record]
        metrics['accuracy'] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0

    # 找出预测性能最佳和最差的记录
    records_by_accuracy = sorted(record_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    best_records = records_by_accuracy[:5]
    worst_records = records_by_accuracy[-5:]

    return {
        'overall': {
            'total_samples': len(y_true),
            'correct_samples': len(correct_indices),
            'error_samples': len(error_indices),
            'accuracy': len(correct_indices) / len(y_true),
            'boundary_samples': len(boundary_indices),
            'false_positive': len(false_positive),
            'false_negative': len(false_negative)
        },
        'record_metrics': record_metrics,
        'best_records': best_records,
        'worst_records': worst_records,
        'indices': {
            'correct': correct_indices,
            'error': error_indices,
            'boundary': boundary_indices,
            'false_positive': false_positive,
            'false_negative': false_negative
        }
    }

def plot_sdb_distribution(record_info, dataset_type='all', save_path=None):
    """
    可视化SDB事件在数据集中的分布

    参数:
    record_info: 记录样本来源信息的列表
    dataset_type: 数据集类型(如'all', 'train', 'valid', 'test')
    save_path: 保存图表的路径
    """
    # 提取SDB比例
    sdb_ratios = [info.get('sdb_ratio', 0) for info in record_info]

    plt.figure(figsize=(10, 6))
    plt.hist(sdb_ratios, bins=50, alpha=0.75)
    plt.xlabel('SDB事件占比')
    plt.ylabel('窗口数量')
    plt.title(f'SDB事件分布 ({dataset_type}数据集)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加SDB阈值线
    plt.axvline(x=0.3, color='r', linestyle='--',
                label='SDB阈值 (30%)')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"SDB分布图已保存到 {save_path}")

    plt.close()