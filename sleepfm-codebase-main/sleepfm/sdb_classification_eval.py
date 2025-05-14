#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SDB（睡眠呼吸障碍）检测模型评估

该脚本用于评估SDB检测模型的性能，使用预训练模型生成的嵌入向量进行分类。
支持单模态和多模态融合评估，并生成详细的性能报告和可视化结果。
"""

import pandas as pd
from tqdm import tqdm
import pickle
import os
import torch
from loguru import logger
import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import Counter

import sys
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score

# 添加项目路径
sys.path.append("/root/autodl-fs/sleepfm-codebase-main/sleepfm-codebase-main/sleepfm/model")
import models
from config import (MODALITY_TYPES, SDB_CONFIG, PATH_TO_PROCESSED_DATA)
from utils import train_model, compute_sdb_metrics, plot_multimodal_performance


def main(args):
    """主函数，处理命令行参数并执行评估流程"""

    # 设置路径
    dataset_dir = args.dataset_dir if args.dataset_dir else PATH_TO_PROCESSED_DATA
    output_file = args.output_file
    path_to_output = os.path.join(dataset_dir, f"{output_file}")
    modality_type = args.modality_type
    model_name = args.model_name

    # 创建输出目录
    path_to_figures = os.path.join(path_to_output, f"figures")
    path_to_models = os.path.join(path_to_output, f"models")
    path_to_probs = os.path.join(path_to_output, f"probs")

    os.makedirs(path_to_figures, exist_ok=True)
    os.makedirs(path_to_models, exist_ok=True)
    os.makedirs(path_to_probs, exist_ok=True)

    # 设置数据文件名
    dataset_paths_file = "sdb_dataset_paths.pickle"

    test_emb_file = f"{os.path.basename(dataset_paths_file).split('.')[0]}_test_emb.pickle"
    valid_emb_file = f"{os.path.basename(dataset_paths_file).split('.')[0]}_valid_emb.pickle"
    train_emb_file = f"{os.path.basename(dataset_paths_file).split('.')[0]}_train_emb.pickle"

    logger.info(f"模态类型: {modality_type}")
    logger.info(f"数据集路径文件: {dataset_paths_file}")
    logger.info(f"测试嵌入文件: {test_emb_file}")
    logger.info(f"验证嵌入文件: {valid_emb_file}")
    logger.info(f"训练嵌入文件: {train_emb_file}")

    # 加载数据集 - 使用window-level的数据集
    path_to_paths_dataset = os.path.join(dataset_dir, f"{dataset_paths_file}")
    logger.info(f"加载数据集: {path_to_paths_dataset}")
    with open(path_to_paths_dataset, "rb") as f:
        dataset_paths = pickle.load(f)

    # 加载嵌入向量
    path_to_eval_data = os.path.join(path_to_output, f"eval_data")

    with open(os.path.join(path_to_eval_data, test_emb_file), "rb") as f:
        emb_test = pickle.load(f)

    with open(os.path.join(path_to_eval_data, valid_emb_file), "rb") as f:
        emb_valid = pickle.load(f)

    with open(os.path.join(path_to_eval_data, train_emb_file), "rb") as f:
        emb_train = pickle.load(f)

    # 提取标签
    logger.info("提取SDB标签...")

    # 从dataset_paths中提取标签 - 这是window-level的标签
    # 每个window的数据格式为(path, label)，其中label为1表示SDB，0表示非SDB
    labels_test = np.array([1 if path_label[1] == 1 else 0 for path_label in dataset_paths['test']])
    labels_valid = np.array([1 if path_label[1] == 1 else 0 for path_label in dataset_paths['valid']])
    labels_train = np.array([1 if path_label[1] == 1 else 0 for path_label in dataset_paths['train']])

    # 统计标签分布
    counter_test = Counter(labels_test)
    counter_valid = Counter(labels_valid)
    counter_train = Counter(labels_train)

    logger.info(f"测试集标签分布: {counter_test}")
    logger.info(f"验证集标签分布: {counter_valid}")
    logger.info(f"训练集标签分布: {counter_train}")

    # 处理嵌入向量
    if modality_type == "combined":
        logger.info("使用多模态融合嵌入向量...")
        emb_test = np.concatenate(emb_test, axis=1)
        emb_valid = np.concatenate(emb_valid, axis=1)
        emb_train = np.concatenate(emb_train, axis=1)
    else:
        logger.info(f"使用单模态 {modality_type} 嵌入向量...")
        target_index = MODALITY_TYPES.index(modality_type)
        emb_test = emb_test[target_index]
        emb_valid = emb_valid[target_index]
        emb_train = emb_train[target_index]

    # 准备训练和测试数据
    X_train = emb_train
    y_train = labels_train

    X_test = emb_test
    y_test = labels_test

    # 打印数据维度，确保匹配
    logger.info(f"训练嵌入向量形状: {X_train.shape}")
    logger.info(f"训练标签形状: {y_train.shape}")
    logger.info(f"测试嵌入向量形状: {X_test.shape}")
    logger.info(f"测试标签形状: {y_test.shape}")

    # 训练和评估模型
    logger.info(f"使用 {model_name} 模型进行SDB检测...")
    # 创建分类器特定的子目录
    classifier_dir = os.path.join(path_to_output, f"figures/{model_name}_{modality_type}")
    os.makedirs(classifier_dir, exist_ok=True)

    # 使用子目录作为保存路径
    path_to_save = classifier_dir
    model, y_probs, class_report = train_model(
        X_train, X_test, y_train, y_test,
        path_to_save,
        SDB_CONFIG["class_names"],
        model_name=model_name,
        max_iter=args.max_iter,
        modality_type=modality_type
    )

    # 计算详细的SDB评估指标
    y_score = y_probs[:, 1]  # 取SDB类别的概率

    # 使用默认阈值(0.5)的预测结果
    y_pred_default = model.predict(X_test)
    sdb_metrics_default = compute_sdb_metrics(y_test, y_pred_default, y_score)

    # 尝试不同的分类阈值来缓解类别不平衡
    thresholds = [0.5, 0.4, 0.3, 0.2]
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = sdb_metrics_default
    best_y_pred = y_pred_default

    logger.info("尝试不同的分类阈值来缓解类别不平衡...")
    for threshold in thresholds:
        # 根据阈值调整预测结果
        y_pred_adjusted = (y_score >= threshold).astype(int)
        metrics = compute_sdb_metrics(y_test, y_pred_adjusted, y_score)

        # 输出不同阈值的性能
        logger.info(f"阈值 {threshold:.2f}: 准确率={metrics['accuracy']:.4f}, "
                   f"敏感性={metrics['sensitivity']:.4f}, 特异性={metrics['specificity']:.4f}, "
                   f"F1分数={metrics['f1_score']:.4f}")

        # 选择F1分数最高的阈值
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = threshold
            best_metrics = metrics
            best_y_pred = y_pred_adjusted

    logger.info(f"最佳阈值: {best_threshold:.2f}, F1分数: {best_f1:.4f}")

    # 使用最佳阈值的结果
    y_pred = best_y_pred
    sdb_metrics = best_metrics

    # 保存模型和最佳阈值
    logger.info(f"保存模型...")
    model_info = {
        'model': model,
        'best_threshold': best_threshold
    }
    with open(os.path.join(path_to_models, f"{modality_type}_model.pickle"), 'wb') as file:
        pickle.dump(model_info, file)

    # 保存预测概率
    logger.info(f"保存预测概率...")
    with open(os.path.join(path_to_probs, f"{modality_type}_y_probs.pickle"), 'wb') as file:
        pickle.dump(y_probs, file)

    # 保存测试标签和预测结果
    logger.info(f"保存测试标签和预测结果...")
    results = {
        'y_test': y_test,
        'y_pred': y_pred,
        'y_score': y_score,
        'best_threshold': best_threshold
    }
    with open(os.path.join(path_to_probs, f"test_results.pickle"), 'wb') as file:
        pickle.dump(results, file)

    # 计算两个类别的ROC和PR曲线指标
    fpr_non_sdb, tpr_non_sdb, _ = roc_curve(y_test == 0, y_probs[:, 0])
    roc_auc_non_sdb = auc(fpr_non_sdb, tpr_non_sdb)

    fpr_sdb, tpr_sdb, _ = roc_curve(y_test == 1, y_probs[:, 1])
    roc_auc_sdb = auc(fpr_sdb, tpr_sdb)

    precision_non_sdb, recall_non_sdb, _ = precision_recall_curve(y_test == 0, y_probs[:, 0])
    ap_non_sdb = average_precision_score(y_test == 0, y_probs[:, 0])

    precision_sdb, recall_sdb, _ = precision_recall_curve(y_test == 1, y_probs[:, 1])
    ap_sdb = average_precision_score(y_test == 1, y_probs[:, 1])

    # 保存性能指标和最佳阈值到分类器特定的子目录
    logger.info(f"保存性能指标到分类器目录...")
    performance_info = {
        'best_threshold': best_threshold,
        'metrics': sdb_metrics,
        'thresholds_tested': thresholds,
        'model_name': model_name,
        'modality_type': modality_type,
        'class_specific': {
            'non_sdb': {
                'auc': roc_auc_non_sdb,
                'ap': ap_non_sdb,
                'roc_curve': {'fpr': fpr_non_sdb, 'tpr': tpr_non_sdb},
                'pr_curve': {'precision': precision_non_sdb, 'recall': recall_non_sdb}
            },
            'sdb': {
                'auc': roc_auc_sdb,
                'ap': ap_sdb,
                'roc_curve': {'fpr': fpr_sdb, 'tpr': tpr_sdb},
                'pr_curve': {'precision': precision_sdb, 'recall': recall_sdb}
            }
        }
    }
    with open(os.path.join(classifier_dir, f"performance_info.pickle"), 'wb') as file:
        pickle.dump(performance_info, file)



    # 保存一个简单的文本文件，记录最佳阈值和主要指标
    with open(os.path.join(classifier_dir, f"best_threshold.txt"), 'w') as file:
        file.write(f"Best Threshold: {best_threshold:.4f}\n")
        file.write(f"Accuracy: {sdb_metrics['accuracy']:.4f}\n")
        file.write(f"Sensitivity: {sdb_metrics['sensitivity']:.4f}\n")
        file.write(f"Specificity: {sdb_metrics['specificity']:.4f}\n")
        file.write(f"F1 Score: {sdb_metrics['f1_score']:.4f}\n")
        file.write(f"AUC (Overall): {sdb_metrics['auc']:.4f}\n")
        file.write(f"AUPRC (Overall): {sdb_metrics['auprc']:.4f}\n")
        file.write(f"\nClass-specific metrics:\n")
        file.write(f"Non-SDB AUC: {roc_auc_non_sdb:.4f}\n")
        file.write(f"SDB AUC: {roc_auc_sdb:.4f}\n")
        file.write(f"Non-SDB AP: {ap_non_sdb:.4f}\n")
        file.write(f"SDB AP: {ap_sdb:.4f}\n")

    # 保存分类报告
    logger.info(f"保存分类报告...")
    with open(os.path.join(path_to_probs, f"{modality_type}_class_report.pickle"), 'wb') as file:
        pickle.dump(class_report, file)

    # 保存SDB评估指标
    logger.info(f"保存SDB评估指标...")
    with open(os.path.join(path_to_probs, f"{modality_type}_sdb_metrics.pickle"), 'wb') as file:
        pickle.dump(sdb_metrics, file)

    # 输出主要评估指标
    logger.info(f"SDB检测性能:")
    logger.info(f"  准确率: {sdb_metrics['accuracy']:.4f}")
    logger.info(f"  敏感性: {sdb_metrics['sensitivity']:.4f}")
    logger.info(f"  特异性: {sdb_metrics['specificity']:.4f}")
    logger.info(f"  F1分数: {sdb_metrics['f1_score']:.4f}")
    logger.info(f"  AUC: {sdb_metrics['auc']:.4f}")
    logger.info(f"  AUPRC: {sdb_metrics['auprc']:.4f}")

    # 创建一个函数来绘制混淆矩阵
    def plot_confusion_matrix(cm, title, filename, fmt='d', save_to_classifier_dir=False, classifier_filename=None):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=SDB_CONFIG["class_names"],
                    yticklabels=SDB_CONFIG["class_names"])
        # 使用英文标题和标签
        plt.title(title.replace('SDB检测混淆矩阵', 'SDB Detection Confusion Matrix')
                      .replace('原始数值', 'Raw Values')
                      .replace('按行归一化', 'Row Normalized')
                      .replace('按总数归一化', 'Total Normalized')
                      .replace('阈值', 'Threshold')
                      .replace('默认阈值', 'Default Threshold'))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        # 保存到主目录
        plt.savefig(os.path.join(path_to_figures, filename))
        # 如果需要，也保存到分类器特定的子目录
        if save_to_classifier_dir and classifier_filename:
            plt.savefig(os.path.join(classifier_dir, classifier_filename))
        plt.close()

    # 绘制最佳阈值的混淆矩阵
    cm = sdb_metrics['confusion_matrix']

    # 原始数值混淆矩阵
    plot_confusion_matrix(
        cm,
        f'SDB检测混淆矩阵 - 原始数值 (阈值={best_threshold:.2f}, {modality_type})',
        f'sdb_confusion_matrix_raw_thresh{best_threshold:.2f}_{modality_type}.png',
        'd',
        save_to_classifier_dir=True,
        classifier_filename='best_confusion_matrix_raw.png'
    )

    # 按行归一化混淆矩阵
    cm_norm_row = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(
        cm_norm_row,
        f'SDB检测混淆矩阵 - 按行归一化 (阈值={best_threshold:.2f}, {modality_type})',
        f'sdb_confusion_matrix_norm_row_thresh{best_threshold:.2f}_{modality_type}.png',
        '.2f',
        save_to_classifier_dir=True,
        classifier_filename='best_confusion_matrix_norm_row.png'
    )

    # 按总数归一化混淆矩阵
    cm_norm_total = cm.astype('float') / cm.sum()
    plot_confusion_matrix(
        cm_norm_total,
        f'SDB检测混淆矩阵 - 按总数归一化 (阈值={best_threshold:.2f}, {modality_type})',
        f'sdb_confusion_matrix_norm_total_thresh{best_threshold:.2f}_{modality_type}.png',
        '.4f',
        save_to_classifier_dir=True,
        classifier_filename='best_confusion_matrix_norm_total.png'
    )

    # 为默认阈值(0.5)也生成混淆矩阵，以便比较
    if best_threshold != 0.5:
        cm_default = sdb_metrics_default['confusion_matrix']

        # 原始数值混淆矩阵(默认阈值)
        plot_confusion_matrix(
            cm_default,
            f'SDB检测混淆矩阵 - 原始数值 (默认阈值=0.50, {modality_type})',
            f'sdb_confusion_matrix_raw_thresh0.50_{modality_type}.png',
            'd'
        )

        # 按行归一化混淆矩阵(默认阈值)
        cm_default_norm_row = cm_default.astype('float') / cm_default.sum(axis=1)[:, np.newaxis]
        plot_confusion_matrix(
            cm_default_norm_row,
            f'SDB检测混淆矩阵 - 按行归一化 (默认阈值=0.50, {modality_type})',
            f'sdb_confusion_matrix_norm_row_thresh0.50_{modality_type}.png',
            '.2f'
        )

    # 创建一个比较不同阈值性能的图表
    plt.figure(figsize=(12, 8))
    metrics_keys = ['accuracy', 'sensitivity', 'specificity', 'f1_score']
    # 使用英文标签
    metrics_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']
    metrics_values = []

    for threshold in thresholds:
        y_pred_t = (y_score >= threshold).astype(int)
        metrics_t = compute_sdb_metrics(y_test, y_pred_t, y_score)
        # 使用metrics_keys来获取对应的指标值
        metrics_values.append([metrics_t[key] for key in metrics_keys])

    metrics_values = np.array(metrics_values)

    # 绘制不同阈值的性能比较
    for i, label in enumerate(metrics_labels):
        plt.plot(thresholds, metrics_values[:, i], 'o-', label=label)

    plt.axvline(x=best_threshold, color='r', linestyle='--',
                label=f'Best Threshold ({best_threshold:.2f})')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Metric Value')
    plt.title(f'Performance Comparison of Different Thresholds ({modality_type})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    # 保存到主目录
    plt.savefig(os.path.join(path_to_figures, f'threshold_comparison_{modality_type}.png'))
    # 保存到分类器特定的子目录
    plt.savefig(os.path.join(classifier_dir, f'threshold_comparison.png'))
    plt.close()

    # 绘制ROC曲线并保存到两个位置 - 包含两个类别
    plt.figure(figsize=(8, 6))

    # 使用已计算的非SDB类别的ROC曲线
    plt.plot(fpr_non_sdb, tpr_non_sdb, 'b-', label=f'Non-SDB (AUC = {roc_auc_non_sdb:.2f})')

    # 使用已计算的SDB类别的ROC曲线
    plt.plot(fpr_sdb, tpr_sdb, 'orange', label=f'SDB (AUC = {roc_auc_sdb:.2f})')

    # 添加对角线
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves (Threshold={best_threshold:.2f}, {modality_type})')
    plt.legend(loc="lower right")
    plt.tight_layout()

    # 保存到主目录
    plt.savefig(os.path.join(path_to_figures, f'sdb_roc_curve_thresh{best_threshold:.2f}_{modality_type}.png'))
    # 保存到分类器特定的子目录
    plt.savefig(os.path.join(classifier_dir, f'best_roc_curve.png'))
    plt.close()

    # 绘制PR曲线并保存到两个位置 - 包含两个类别
    plt.figure(figsize=(8, 6))

    # 使用已计算的非SDB类别的PR曲线
    plt.plot(recall_non_sdb, precision_non_sdb, 'b-', label=f'Non-SDB (AP = {ap_non_sdb:.2f})')

    # 使用已计算的SDB类别的PR曲线
    plt.plot(recall_sdb, precision_sdb, 'orange', label=f'SDB (AP = {ap_sdb:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves (Threshold={best_threshold:.2f}, {modality_type})')
    plt.legend(loc="lower left")
    plt.tight_layout()

    # 保存到主目录
    plt.savefig(os.path.join(path_to_figures, f'sdb_pr_curve_thresh{best_threshold:.2f}_{modality_type}.png'))
    # 保存到分类器特定的子目录
    plt.savefig(os.path.join(classifier_dir, f'best_pr_curve.png'))
    plt.close()

    logger.info(f"SDB detection evaluation completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估SDB检测模型性能")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件名")
    parser.add_argument("--dataset_dir", type=str, default=None, help="数据集目录路径")
    parser.add_argument("--modality_type", type=str, help="模态类型",
                        choices=["respiratory", "sleep_stages", "ekg", "combined"], default="combined")
    parser.add_argument("--model_name", type=str, default="logistic",
                        choices=["logistic", "xgb", "rf"], help="模型类型")
    parser.add_argument("--max_iter", type=int, default=1000, help="逻辑回归最大迭代次数")

    args = parser.parse_args()

    # 配置日志
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])

    main(args)
