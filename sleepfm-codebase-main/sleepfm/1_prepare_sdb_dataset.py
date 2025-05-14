#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
准备SDB（睡眠呼吸障碍）检测数据集

该脚本用于将提取的SDB数据划分为训练集、验证集和测试集，
适用于二分类任务（SDB检测）。
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
import argparse
from loguru import logger
from tqdm import tqdm
import multiprocessing
import random
import sys
from config import SDB_CONFIG

def parallel_prepare_data(args):
    """
    并行处理数据，将数据按照患者ID划分到不同的数据集中

    参数:
        args: 包含以下元素的元组:
            mrns: 患者ID列表
            dataset_dir: 数据集目录
            mrn_pretrain: 预训练集患者ID
            mrn_train: 训练集患者ID
            mrn_valid: 验证集患者ID
            mrn_test: 测试集患者ID

    返回:
        data_dict: 包含划分后数据的字典
    """
    mrns = args[0]
    dataset_dir = args[1]
    mrn_pretrain = args[2]
    mrn_train = args[3]
    mrn_valid = args[4]
    mrn_test = args[5]

    data_dict = {
        "pretrain": [],
        "train": [],
        "valid": [],
        "test": []
    }

    empty_label_dict_counts = 0
    path_to_Y = os.path.join(dataset_dir, "Y")
    for mrn in tqdm(mrns):
        one_patient_dict = {
            mrn: {}
        }

        path_to_X = os.path.join(dataset_dir, "X")
        path_to_patient = os.path.join(path_to_X, mrn)
        path_to_label = os.path.join(path_to_Y, f"{mrn}.pickle")

        if mrn in mrn_pretrain:
            split_name = "pretrain"
        elif mrn in mrn_train:
            split_name = "train"
        elif mrn in mrn_valid:
            split_name = "valid"
        elif mrn in mrn_test:
            split_name = "test"
        else:
            logger.warning(f"{mrn} 不在任何数据集划分中")
            continue

        if os.path.exists(path_to_label):
            with open(path_to_label, 'rb') as file:
                labels_dict = pickle.load(file)
        else:
            logger.info(f"{mrn} 标签文件不存在")
            continue

        if len(labels_dict) == 0:
            logger.info(f"{mrn} 标签字典为空")
            empty_label_dict_counts += 1
            continue

        if not os.path.exists(path_to_patient):
            logger.info(f"{mrn} 数据不存在")
            continue

        # 对于SDB检测，我们有两个类别：0（非SDB）和1（SDB）
        for event_data_name in os.listdir(path_to_patient):
            event_data_path = os.path.join(path_to_patient, event_data_name)

            # 获取标签（0或1）
            label = labels_dict[event_data_name]

            # 确保标签是整数
            if isinstance(label, dict):
                label = list(label.keys())[0]

            # 将标签转换为整数（确保是0或1）
            label = int(label)

            if label not in one_patient_dict[mrn]:
                one_patient_dict[mrn][label] = []

            one_patient_dict[mrn][label].append(event_data_path)

        data_dict[split_name].append(one_patient_dict)

    logger.info(f"总空标签字典数: {empty_label_dict_counts}")

    return data_dict

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="处理SDB数据并创建数据集")
    parser.add_argument("--dataset_dir", type=str, default=None, help="数据目录路径")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--pretrain_size", type=float, default=0.6, help="预训练集比例")
    parser.add_argument("--train_size", type=float, default=0.2, help="训练集比例")
    parser.add_argument("--valid_size", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_size", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--num_threads", type=int, default=4, help="并行处理线程数")
    parser.add_argument("--min_sample", type=int, default=-1, help="每个类别的最小样本数，-1表示使用所有样本")
    parser.add_argument("--balance_ratio", type=float, default=1.0,
                        help="非SDB样本与SDB样本的比例，用于处理类别不平衡问题")

    args = parser.parse_args()

    # 设置数据集目录
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        dataset_dir = SDB_CONFIG["data_path"]

    random_state = args.random_state
    num_threads = args.num_threads
    pretrain_size = args.pretrain_size
    train_size = args.train_size
    valid_size = args.valid_size
    test_size = args.test_size
    balance_ratio = args.balance_ratio

    # 验证比例总和是否为1
    total_ratio = pretrain_size + train_size + valid_size + test_size
    if abs(total_ratio - 1.0) > 1e-6:
        logger.warning(f"数据集比例总和不为1: {total_ratio}")
        logger.warning(f"将按照比例自动调整")
        # 归一化比例
        pretrain_size /= total_ratio
        train_size /= total_ratio
        valid_size /= total_ratio
        test_size /= total_ratio
        logger.info(f"调整后的比例: pretrain={pretrain_size:.2f}, train={train_size:.2f}, valid={valid_size:.2f}, test={test_size:.2f}")

    logger.info(f"使用数据集目录: {dataset_dir}")

    # 获取所有患者ID
    path_to_X = os.path.join(dataset_dir, "X")
    if not os.path.exists(path_to_X):
        logger.error(f"数据目录不存在: {path_to_X}")
        logger.info("尝试使用默认路径: /root/autodl-tmp/processed_data/pc18/sdb/X")
        path_to_X = "/root/autodl-tmp/processed_data/pc18/sdb/X"
        dataset_dir = "/root/autodl-tmp/processed_data/pc18/sdb"
        if not os.path.exists(path_to_X):
            logger.error(f"默认数据目录也不存在: {path_to_X}")
            return

    mrns = os.listdir(path_to_X)

    if args.debug:
        logger.info("运行调试模式")
        mrns = mrns[:100]
    logger.info(f"处理的患者数量: {len(mrns)}")

    # 划分数据集
    # 按照指定比例划分为pretrain/train/valid/test
    # 首先，将数据集分为pretrain和其他
    remaining_ratio = train_size + valid_size + test_size
    mrn_pretrain, mrn_remaining = train_test_split(mrns, test_size=remaining_ratio, random_state=random_state)

    # 然后，将剩余的部分分为train和valid+test
    valid_test_ratio = (valid_size + test_size) / remaining_ratio
    mrn_train, mrn_valid_test = train_test_split(mrn_remaining, test_size=valid_test_ratio, random_state=random_state)

    # 最后，将valid+test分为valid和test
    test_ratio = test_size / (valid_size + test_size)
    mrn_valid, mrn_test = train_test_split(mrn_valid_test, test_size=test_ratio, random_state=random_state)

    # 转换为集合以加速查找
    mrn_pretrain = set(mrn_pretrain)
    mrn_train = set(mrn_train)
    mrn_valid = set(mrn_valid)
    mrn_test = set(mrn_test)

    # 计算实际的数据集大小和比例
    total_samples = len(mrns)
    pretrain_count = len(mrn_pretrain)
    train_count = len(mrn_train)
    valid_count = len(mrn_valid)
    test_count = len(mrn_test)

    pretrain_ratio = pretrain_count / total_samples
    train_ratio = train_count / total_samples
    valid_ratio = valid_count / total_samples
    test_ratio = test_count / total_samples

    logger.info(f"预训练集: {pretrain_count}个样本 ({pretrain_ratio:.2%})")
    logger.info(f"训练集: {train_count}个样本 ({train_ratio:.2%})")
    logger.info(f"验证集: {valid_count}个样本 ({valid_ratio:.2%})")
    logger.info(f"测试集: {test_count}个样本 ({test_ratio:.2%})")
    logger.info(f"总样本数: {total_samples}个")

    # 并行处理数据
    mrns_per_thread = np.array_split(mrns, num_threads)
    tasks = [(mrns_one_thread, dataset_dir, mrn_pretrain, mrn_train, mrn_valid, mrn_test) for mrns_one_thread in mrns_per_thread]

    with multiprocessing.Pool(num_threads) as pool:
        preprocessed_results = list(pool.imap_unordered(parallel_prepare_data, tasks))

    # 合并结果
    dataset = {}
    for data_dict in preprocessed_results:
        for key, value in data_dict.items():
            if key not in dataset:
                dataset[key] = value
            else:
                dataset[key].extend(value)

    # 按患者ID排序
    for key in dataset:
        dataset[key] = sorted(dataset[key], key=lambda x: list(x.keys())[0])

    # 保存完整数据集
    output_dir = dataset_dir
    logger.info(f"保存数据集到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = os.path.join(output_dir, "sdb_dataset.pickle")
    with open(dataset_path, 'wb') as file:
        pickle.dump(dataset, file)
    logger.info(f"完整数据集已保存到: {dataset_path}")

    # 创建事件级数据集
    dataset_event = {}
    for split, split_data in tqdm(dataset.items(), total=len(dataset)):
        sampled_data = []
        sdb_samples = []
        non_sdb_samples = []

        # 收集所有SDB和非SDB样本
        for item in split_data:
            mrn = list(item.keys())[0]
            patient_data = item[mrn]

            # 收集SDB样本
            if 1 in patient_data:
                sdb_events = patient_data[1]
                if args.min_sample == -1:
                    sampled_sdb = sdb_events
                else:
                    random.seed(args.random_state)
                    sampled_sdb = random.sample(sdb_events, min(args.min_sample, len(sdb_events)))
                sdb_samples.extend([(path, 1) for path in sampled_sdb])

            # 收集非SDB样本
            if 0 in patient_data:
                non_sdb_events = patient_data[0]
                if args.min_sample == -1:
                    sampled_non_sdb = non_sdb_events
                else:
                    random.seed(args.random_state)
                    sampled_non_sdb = random.sample(non_sdb_events, min(args.min_sample, len(non_sdb_events)))
                non_sdb_samples.extend([(path, 0) for path in sampled_non_sdb])

        # 平衡类别（如果需要）
        if balance_ratio != 1.0 and len(sdb_samples) > 0:
            target_non_sdb_count = int(len(sdb_samples) * balance_ratio)
            if len(non_sdb_samples) > target_non_sdb_count:
                random.seed(args.random_state)
                non_sdb_samples = random.sample(non_sdb_samples, target_non_sdb_count)

        # 合并样本
        sampled_data = sdb_samples + non_sdb_samples

        # 打乱数据
        random.seed(args.random_state)
        random.shuffle(sampled_data)

        # 记录类别统计信息
        sdb_count = sum(1 for _, label in sampled_data if label == 1)
        non_sdb_count = sum(1 for _, label in sampled_data if label == 0)
        logger.info(f"{split} 集: 总样本数 = {len(sampled_data)}, SDB样本数 = {sdb_count}, 非SDB样本数 = {non_sdb_count}")

        dataset_event[split] = sampled_data

    # 保存事件级数据集
    events_path = os.path.join(output_dir, "sdb_dataset_paths.pickle")
    with open(events_path, 'wb') as file:
        pickle.dump(dataset_event, file)
    logger.info(f"事件级数据集已保存到: {events_path}")

    logger.info("数据集准备完成!")
    logger.info(f"您可以使用以下文件进行模型训练:")
    logger.info(f"1. 完整数据集: {dataset_path}")
    logger.info(f"2. 事件级数据集: {events_path}")

if __name__ == "__main__":
    # 配置日志
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])

    main()
