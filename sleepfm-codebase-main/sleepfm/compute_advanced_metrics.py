import pickle
import numpy as np
import os
from collections import Counter
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
import pandas as pd
from config import CLASS_LABELS

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
        f1 = f1_score(y_true == i, y_pred == i)
        
        # Bootstrap采样计算置信区间
        indices = np.arange(len(y_true))
        for _ in range(n_bootstrap):
            bootstrap_indices = resample(indices, replace=True)
            
            if len(np.unique(y_true[bootstrap_indices])) < 2:
                continue  # 跳过只有一个类别的样本
                
            try:
                auroc_bootstrap = roc_auc_score(y_true_bin[bootstrap_indices, i], y_probs[bootstrap_indices, i])
                auprc_bootstrap = average_precision_score(y_true_bin[bootstrap_indices, i], y_probs[bootstrap_indices, i])
                f1_bootstrap = f1_score(y_true[bootstrap_indices] == i, y_pred[bootstrap_indices] == i)
                
                auroc_samples.append(auroc_bootstrap)
                auprc_samples.append(auprc_bootstrap)
                f1_samples.append(f1_bootstrap)
            except:
                continue
            
        # 计算置信区间
        alpha = (1 - ci) / 2
        auroc_ci = np.quantile(auroc_samples, [alpha, 1-alpha])
        auprc_ci = np.quantile(auprc_samples, [alpha, 1-alpha])
        f1_ci = np.quantile(f1_samples, [alpha, 1-alpha])
        
        # 计算标准差（论文中使用的是±标准差形式）
        auroc_std = np.std(auroc_samples)
        auprc_std = np.std(auprc_samples)
        f1_std = np.std(f1_samples)
        
        metrics_dict[label] = {
            'AUROC': {'value': auroc, 'std': auroc_std, 'ci': auroc_ci},
            'AUPRC': {'value': auprc, 'std': auprc_std, 'ci': auprc_ci},
            'F1': {'value': f1, 'std': f1_std, 'ci': f1_ci}
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

def main():
    # 路径配置
    base_dir = "/root/autodl-tmp/processed_data/pc18/outputs/output_pairwise_dataset_events_-1_lr_0.01_lr_sp_5_wd_0.0_bs_32_respiratory_sleep_stages_ekg"
    probs_path = os.path.join(base_dir, "probs/combined_y_probs.pickle")
    
    # 还需要真实标签
    # 注: 这里需要加载测试集的真实标签，根据您的数据组织方式可能需要调整
    # 假设有一个文件保存了测试集真实标签
    labels_path = os.path.join(base_dir, "probs/test_labels.pickle")  # 这个路径可能需要调整
    
    try:
        # 加载预测概率
        print(f"正在加载预测概率文件: {probs_path}")
        with open(probs_path, "rb") as f:
            y_probs = pickle.load(f)
        
        # 加载真实标签 (需要确保这个文件存在，或者通过其他方式获取标签)
        print(f"尝试加载测试集标签: {labels_path}")
        try:
            with open(labels_path, "rb") as f:
                y_true = pickle.load(f)
                
            # 检查标签与预测概率的样本数量是否匹配
            if len(y_true) != y_probs.shape[0]:
                print(f"警告: 测试标签数量({len(y_true)})与预测概率样本数量({y_probs.shape[0]})不匹配")
                
                # 如果class_report文件存在，尝试从中获取标签和预测结果
                class_report_path = os.path.join(base_dir, "probs", f"combined_class_report.pickle")
                if os.path.exists(class_report_path):
                    print(f"尝试从分类报告中提取信息...")
                    with open(class_report_path, "rb") as f:
                        class_report = pickle.load(f)
                    
                    if isinstance(class_report, dict) and 'accuracy' in class_report:
                        accuracy = class_report.pop('accuracy', None)
                        macro_avg = class_report.pop('macro avg', None)
                        weighted_avg = class_report.pop('weighted avg', None)
                        
                        print(f"\n从分类报告中提取的结果:")
                        print(f"整体准确率: {accuracy:.4f}")
                        
                        print("\n各睡眠阶段的详细指标:")
                        df = pd.DataFrame(class_report).T
                        df = df.sort_index()
                        print(df.round(4))
                        
                        print("\n宏平均(各类别平均)指标:")
                        for metric, value in macro_avg.items():
                            print(f"  {metric}: {value:.4f}")
                        
                        print("\n加权平均(考虑各类别样本数)指标:")
                        for metric, value in weighted_avg.items():
                            print(f"  {metric}: {value:.4f}")
                        
                        return
                
                # 如果无法从分类报告中获取信息，则终止
                print("无法计算指标，请确保测试标签与预测概率样本数量匹配")
                return
        except FileNotFoundError:
            print(f"警告: 找不到测试集标签文件，请提供真实标签才能计算指标")
            print("您可以尝试重新运行评估脚本，确保保存测试集标签")
            return
        
        # 计算指标
        print("计算AUROC、AUPRC和F1指标...")
        metrics = compute_metrics_with_ci(y_true, y_probs)
        
        # 显示结果
        print("\nSleepFM在CinC数据集上的性能:")
        print("-" * 80)
        print(f"{'睡眠阶段':<10} | {'AUROC':<15} | {'AUPRC':<15} | {'F1':<15}")
        print("-" * 80)
        
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
            
            print(f"{label_name:<10} | {auroc:<15} | {auprc:<15} | {f1:<15}")
        
        print("-" * 80)
        
        # 保存结果到文件
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
        
        results_path = os.path.join(base_dir, "advanced_metrics.csv")
        results_df.to_csv(results_path, index=False)
        print(f"结果已保存至: {results_path}")
        
    except Exception as e:
        print(f"评估过程中出错: {e}")

if __name__ == "__main__":
    main() 