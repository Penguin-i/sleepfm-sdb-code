#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成SDB（睡眠呼吸障碍）预训练模型的嵌入向量

该脚本用于加载预训练好的SDB模型，并生成各个模态（呼吸、睡眠阶段、心电图）的嵌入向量。
生成的嵌入向量保存为pickle文件，可用于后续的分类任务评估。
"""

import time
import torch
import os
import click
import tqdm
import math
import pickle
from loguru import logger
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from sleepfm.model import models
    from sleepfm.config import (CONFIG, CHANNEL_DATA,
                        ALL_CHANNELS, CHANNEL_DATA_IDS,
                        PATH_TO_PROCESSED_DATA)
    from sleepfm.model.dataset import EventDataset
except ImportError:
    # 尝试相对导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import models
    from config import (CONFIG, CHANNEL_DATA,
                    ALL_CHANNELS, CHANNEL_DATA_IDS,
                    PATH_TO_PROCESSED_DATA)
    from model.dataset import EventDataset


class StringListParamType(click.ParamType):
    name = 'string_list'

    def convert(self, value, param, ctx):
        if value is None:
            return []
        return value.split(',')


@click.command("generate_eval_embed")
@click.option("--pretrain_dir", type=str,
              default='/root/autodl-tmp/processed_data/pc18/sdb/outputs/sdb_pretrain_leave_one_out_sdb_dataset_paths_lr_0.001_lr_sp_5_wd_0.01_bs_32_respiratory_sleep_stages_ekg_resp_w1.3',
              help='预训练模型目录')
@click.option("--dataset_dir", type=str, default='/root/autodl-tmp/processed_data/pc18/sdb')
@click.option("--dataset_file", type=str, default="sdb_dataset_paths.pickle")
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=4)
@click.option("--splits", type=str, default='train,valid,test', help='指定数据集分割（train, valid, test）')
@click.option("--modality_types", type=StringListParamType(), default="respiratory,sleep_stages,ekg")
def generate_eval_embed(
    pretrain_dir,
    dataset_dir,
    dataset_file,
    batch_size,
    num_workers,
    splits,
    modality_types
):
    """生成预训练模型的评估嵌入向量"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 解析数据集分割
    splits = splits.split(",")
    logger.info(f"将为以下分割生成嵌入向量: {splits}")
    logger.info(f"模态类型: {modality_types}")

    # 确保预训练目录存在
    if not os.path.exists(pretrain_dir):
        logger.error(f"预训练目录不存在: {pretrain_dir}")
        return

    # 构建数据集路径
    path_to_data = os.path.join(dataset_dir, dataset_file)
    logger.info(f"数据集路径: {path_to_data}")

    # 检查数据集文件是否存在
    if not os.path.exists(path_to_data):
        logger.error(f"数据集文件不存在: {path_to_data}")
        # 尝试寻找替代文件
        potential_files = ["sdb_dataset_paths.pickle", "sdb_dataset.pickle", "dataset_events.pkl", "dataset_events_-1.pickle"]
        found = False
        for pot_file in potential_files:
            try_path = os.path.join(dataset_dir, pot_file)
            if os.path.exists(try_path):
                logger.info(f"找到替代数据集文件: {try_path}")
                path_to_data = try_path
                found = True
                break
        if not found:
            logger.error("无法找到有效的数据集文件，程序终止")
            return

    # 加载数据集
    try:
        dataset = {
            split: EventDataset(path_to_data, split=split, modality_type=modality_types)
            for split in splits
        }

        for split in splits:
            logger.info(f"{split} 集大小: {len(dataset[split])}")
    except Exception as e:
        logger.error(f"加载数据集时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # 创建模型
    model_dict = {}

    if "respiratory" in modality_types:
        in_channel = len(CHANNEL_DATA_IDS["Respiratory"])
        model_resp = models.EffNet(in_channel=in_channel, stride=2, dilation=1)
        model_resp.fc = torch.nn.Linear(model_resp.fc.in_features, 512)
        if device.type == "cuda":
            model_resp = torch.nn.DataParallel(model_resp)
        model_resp.to(device)
        model_dict["respiratory"] = model_resp
        logger.info(f"创建呼吸模型: in_channel={in_channel}")

    if "sleep_stages" in modality_types:
        in_channel = len(CHANNEL_DATA_IDS["Sleep_Stages"])
        model_sleep = models.EffNet(in_channel=in_channel, stride=2, dilation=1)
        model_sleep.fc = torch.nn.Linear(model_sleep.fc.in_features, 512)
        if device.type == "cuda":
            model_sleep = torch.nn.DataParallel(model_sleep)
        model_sleep.to(device)
        model_dict["sleep_stages"] = model_sleep
        logger.info(f"创建睡眠阶段模型: in_channel={in_channel}")

    if "ekg" in modality_types:
        in_channel = len(CHANNEL_DATA_IDS["EKG"])
        model_ekg = models.EffNet(in_channel=in_channel, stride=2, dilation=1)
        model_ekg.fc = torch.nn.Linear(model_ekg.fc.in_features, 512)
        if device.type == "cuda":
            model_ekg = torch.nn.DataParallel(model_ekg)
        model_ekg.to(device)
        model_dict["ekg"] = model_ekg
        logger.info(f"创建心电图模型: in_channel={in_channel}")

    # 加载模型权重
    checkpoint_path = os.path.join(pretrain_dir, "best.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(pretrain_dir, "checkpoint.pt")

    if not os.path.exists(checkpoint_path):
        logger.error(f"在 {pretrain_dir} 中未找到模型文件 best.pt 或 checkpoint.pt")
        return

    logger.info(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    temperature = checkpoint.get("temperature", 0.0)
    logger.info(f"预训练模型温度参数: {temperature:.3f}")

    # 加载每个模型的权重
    for model_key in model_dict:
        model_dict[model_key].load_state_dict(checkpoint[f"{model_key}_state_dict"])
        model_dict[model_key].eval()
        logger.info(f"加载 {model_key} 模型权重成功")

    # 创建保存路径
    path_to_save = os.path.join(pretrain_dir, "eval_data")
    os.makedirs(path_to_save, exist_ok=True)
    logger.info(f"嵌入向量将保存到: {path_to_save}")

    # 为每个分割生成嵌入向量
    for split in splits:
        logger.info(f"为 {split} 集生成嵌入向量")
        dataloader = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False
        )

        # 初始化嵌入向量列表
        emb = [[] for _ in range(len(modality_types))]

        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader), desc=f"嵌入向量 ({split})") as pbar:
                for data in dataloader:
                    # 根据模态类型数量处理数据
                    if len(modality_types) == 3:
                        resp, sleep, ekg = data
                        resp = resp.to(device, dtype=torch.float)
                        sleep = sleep.to(device, dtype=torch.float)
                        ekg = ekg.to(device, dtype=torch.float)

                        if "respiratory" in modality_types:
                            idx = modality_types.index("respiratory")
                            emb[idx].append(torch.nn.functional.normalize(model_dict["respiratory"](resp)).cpu())

                        if "sleep_stages" in modality_types:
                            idx = modality_types.index("sleep_stages")
                            emb[idx].append(torch.nn.functional.normalize(model_dict["sleep_stages"](sleep)).cpu())

                        if "ekg" in modality_types:
                            idx = modality_types.index("ekg")
                            emb[idx].append(torch.nn.functional.normalize(model_dict["ekg"](ekg)).cpu())

                    elif "respiratory" in modality_types and "sleep_stages" in modality_types:
                        resp, sleep = data
                        resp = resp.to(device, dtype=torch.float)
                        sleep = sleep.to(device, dtype=torch.float)

                        if "respiratory" in modality_types:
                            idx = modality_types.index("respiratory")
                            emb[idx].append(torch.nn.functional.normalize(model_dict["respiratory"](resp)).cpu())

                        if "sleep_stages" in modality_types:
                            idx = modality_types.index("sleep_stages")
                            emb[idx].append(torch.nn.functional.normalize(model_dict["sleep_stages"](sleep)).cpu())

                    elif "respiratory" in modality_types and "ekg" in modality_types:
                        resp, ekg = data
                        resp = resp.to(device, dtype=torch.float)
                        ekg = ekg.to(device, dtype=torch.float)

                        if "respiratory" in modality_types:
                            idx = modality_types.index("respiratory")
                            emb[idx].append(torch.nn.functional.normalize(model_dict["respiratory"](resp)).cpu())

                        if "ekg" in modality_types:
                            idx = modality_types.index("ekg")
                            emb[idx].append(torch.nn.functional.normalize(model_dict["ekg"](ekg)).cpu())

                    elif "sleep_stages" in modality_types and "ekg" in modality_types:
                        sleep, ekg = data
                        sleep = sleep.to(device, dtype=torch.float)
                        ekg = ekg.to(device, dtype=torch.float)

                        if "sleep_stages" in modality_types:
                            idx = modality_types.index("sleep_stages")
                            emb[idx].append(torch.nn.functional.normalize(model_dict["sleep_stages"](sleep)).cpu())

                        if "ekg" in modality_types:
                            idx = modality_types.index("ekg")
                            emb[idx].append(torch.nn.functional.normalize(model_dict["ekg"](ekg)).cpu())

                    pbar.update()

        # 合并每个模态的嵌入向量
        emb = [torch.cat(modal_emb, dim=0) for modal_emb in emb]

        # 保存嵌入向量
        dataset_prefix = os.path.basename(path_to_data).split(".")[0]
        save_path = os.path.join(path_to_save, f"{dataset_prefix}_{split}_emb.pickle")
        with open(save_path, 'wb') as file:
            pickle.dump(emb, file)

        logger.info(f"已保存 {split} 集嵌入向量到: {save_path}")
        # 打印每个嵌入向量的形状
        for i, modal_type in enumerate(modality_types):
            logger.info(f"{modal_type} 嵌入向量形状: {emb[i].shape}")

    logger.info("所有嵌入向量生成完成")


if __name__ == '__main__':
    # 配置日志
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])

    generate_eval_embed()