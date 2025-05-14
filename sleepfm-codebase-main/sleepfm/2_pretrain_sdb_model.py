import time
import torch
import torchvision
import os
import click
import tqdm
import math
import shutil
import datetime
import numpy as np
from loguru import logger
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sleepfm.model import models
from sleepfm.dataset import EventDataset
from sleepfm.config import (CONFIG, CHANNEL_DATA,
                    ALL_CHANNELS, CHANNEL_DATA_IDS,
                    PATH_TO_PROCESSED_DATA, SDB_CONFIG)


class StringListParamType(click.ParamType):
    name = 'string_list'

    def convert(self, value, param, ctx):
        if value is None:
            return []
        return value.split(',')


@click.command("pretrain")
@click.option("--dataset_dir", type=str, default="/root/autodl-tmp/processed_data/pc18/sdb")
@click.option("--dataset_file", type=str, default="sdb_dataset_paths.pickle")
@click.option("--batch_size", type=int, default=32)
@click.option("--num_workers", type=int, default=4)
@click.option("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数，可以模拟更大的批次大小")
@click.option("--weight_decay", type=float, default=0.01)
@click.option("--lr", type=float, default=0.001)
@click.option("--lr_step_period", type=int, default=5)
@click.option("--epochs", type=int, default=20)
@click.option("--mode", type=click.Choice(["pairwise", "leave_one_out"]), default="leave_one_out")
@click.option("--modality_types", type=StringListParamType(), default="respiratory,sleep_stages,ekg")
@click.option("--respiratory_weight", type=float, default=1.3, help="权重因子，增加呼吸信号在leave_one_out模式中的重要性")
@click.option("--optimizer", type=click.Choice(["adam", "sgd"]), default="adam", help="优化器类型")
@click.option("--adaptive_temp", is_flag=True, default=True, help="是否使用自适应温度调整")
@click.option("--initial_temp", type=float, default=0.07, help="初始温度值")
@click.option("--min_temp", type=float, default=0.01, help="最小温度值")
@click.option("--max_temp", type=float, default=0.5, help="最大温度值")
@click.option("--temp_adjust_factor", type=float, default=0.05, help="温度调整因子，值越大调整越快")
@click.option("--dynamic_weight", is_flag=True, default=True, help="是否动态调整模态权重")
def pretrain(
    dataset_dir,
    dataset_file,
    batch_size,
    num_workers,
    gradient_accumulation_steps,
    weight_decay,
    lr,
    lr_step_period,
    epochs,
    mode,
    modality_types,
    respiratory_weight,
    optimizer,
    adaptive_temp,
    initial_temp,
    min_temp,
    max_temp,
    temp_adjust_factor,
    dynamic_weight
):
    if dataset_dir == None:
        dataset_dir = PATH_TO_PROCESSED_DATA

    dataset_file_prefix = dataset_file.split(".")[0]
    modality_types_string = "_".join(modality_types)
    output = os.path.join(dataset_dir, f"outputs/sdb_pretrain_{mode}_{dataset_file_prefix}_lr_{lr}_lr_sp_{lr_step_period}_wd_{weight_decay}_bs_{batch_size}_{modality_types_string}")

    # 添加呼吸权重标记
    if "respiratory" in modality_types and respiratory_weight > 1.0:
        output += f"_resp_w{respiratory_weight}"

    output = os.path.join(CONFIG.OUTPUT, output)
    os.makedirs(output, exist_ok=True)

    # 初始化温度参数（使用标准SimCLR写法）
    temperature = torch.nn.Parameter(torch.tensor(initial_temp))

    # 在训练开始前清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    logger.info(f"modality_types: {modality_types}")
    logger.info(f"Path to dataset: {dataset_dir}")
    logger.info(f"Path to output: {output}")
    logger.info(f"Training Model: {mode}")
    if "respiratory" in modality_types:
        logger.info(f"Respiratory Weight: {respiratory_weight}")

    logger.info(f"Batch Size: {batch_size}; Gradient Accumulation Steps: {gradient_accumulation_steps}; Effective Batch Size: {batch_size * gradient_accumulation_steps}")
    logger.info(f"Number of Workers: {num_workers}")
    logger.info(f"Weight Decay: {weight_decay}; Learning Rate: {lr}; Learning Step Period: {lr_step_period}")
    logger.info(f"Optimizer: {optimizer}; Adaptive Temperature: {adaptive_temp}")
    logger.info(f"Temperature: initial={initial_temp}, min={min_temp}, max={max_temp}, adjust_factor={temp_adjust_factor}")
    logger.info(f"Dynamic Weight Adjustment: {dynamic_weight}")
    logger.info("使用标准SimCLR温度处理方式: logits = similarity / temperature")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device set to {device}")

    num_targets = len(modality_types)
    ij = sum([((i, j), (j, i)) for i in range(len(modality_types)) for j in range(i + 1, len(modality_types))], ())

    start = time.time()
    path_to_dataset = os.path.join(dataset_dir, dataset_file)

    # 检查数据集文件是否存在
    if not os.path.exists(path_to_dataset):
        logger.error(f"数据集文件不存在: {path_to_dataset}")
        logger.info("请先运行1_prepare_sdb_dataset.py生成数据集")
        return

    # 加载数据集 - 使用pretrain数据集进行预训练，而不是train数据集
    # 逐个加载数据集，避免同时加载所有数据集导致内存峰值过高
    dataset = {}
    for split in ["pretrain", "valid", "test"]:
        logger.info(f"Loading {split} dataset...")
        dataset[split] = EventDataset(path_to_dataset, split=split, modality_type=modality_types)
        # 每加载一个数据集后清理一次内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    logger.info(f"Dataset loaded in {time.time() - start:.1f} seconds")

    # 记录各数据集大小
    for split in dataset:
        logger.info(f"{split} 集大小: {len(dataset[split])}")

    model_dict = {}

    if "respiratory" in modality_types:
        model_resp = models.EffNet(in_channel=len(CHANNEL_DATA_IDS["Respiratory"]), stride=2, dilation=1)
        model_resp.fc = torch.nn.Linear(model_resp.fc.in_features, 512)
        if device.type == "cuda":
            model_resp = torch.nn.DataParallel(model_resp)
        model_resp.to(device)
        model_dict["respiratory"] = model_resp

    if "sleep_stages" in modality_types:
        model_sleep = models.EffNet(in_channel=len(CHANNEL_DATA_IDS["Sleep_Stages"]), stride=2, dilation=1)
        model_sleep.fc = torch.nn.Linear(model_sleep.fc.in_features, 512)
        if device.type == "cuda":
            model_sleep = torch.nn.DataParallel(model_sleep)
        model_sleep.to(device)
        model_dict["sleep_stages"] = model_sleep

    if "ekg" in modality_types:
        model_ekg = models.EffNet(in_channel=len(CHANNEL_DATA_IDS["EKG"]), stride=2, dilation=1)
        model_ekg.fc = torch.nn.Linear(model_ekg.fc.in_features, 512)
        if device.type == "cuda":
            model_ekg = torch.nn.DataParallel(model_ekg)
        model_ekg.to(device)
        model_dict["ekg"] = model_ekg

    optim_params = []
    for model_key, model in model_dict.items():
        optim_params += list(model.parameters())

    optim_params.append(temperature)  # Append the temperature parameter

    # 根据选择创建优化器
    if optimizer == "adam":
        optim = torch.optim.Adam(
            optim_params,
            lr=lr,
            weight_decay=weight_decay
        )
        logger.info("使用Adam优化器")
    else:  # sgd
        optim = torch.optim.SGD(
            optim_params,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
        logger.info("使用SGD优化器")

    if lr_step_period is None:
        lr_step_period = math.inf

    # 使用ReduceLROnPlateau调度器，在验证损失停止改善时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=3, verbose=True
    )
    logger.info("使用ReduceLROnPlateau学习率调度器")

    epoch_resume = 0
    best_loss = math.inf

    if os.path.isfile(os.path.join(output, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))

        for model_key, model in model_dict.items():
            model.load_state_dict(checkpoint[f"{model_key}_state_dict"])

        # Loading temperature and other checkpointed parameters
        with torch.no_grad():
            temperature.fill_(checkpoint["temperature"])
        optim.load_state_dict(checkpoint["optim_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_dict"])

        # Other checkpointed values
        epoch_resume = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        logger.info(f"Resuming from epoch {epoch_resume}\n")
    else:
        logger.info("Starting from scratch")

    # 在训练开始前初始化优化器
    optim.zero_grad()

    os.makedirs(os.path.join(output, "log"), exist_ok=True)
    with open(os.path.join(output, "log", "{}.tsv".format(datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))), "w") as f:
        f.write("Epoch\tSplit\tTotal Loss\t")
        if mode == "pairwise":
            f.write("".join(f"{modality_types[i]}-{modality_types[j]} Loss\t" for (i, j) in ij))
            f.write("".join(f"{modality_types[i]}-{modality_types[j]} Accuracy\t" for (i, j) in ij))
        elif mode == "leave_one_out":
            f.write("".join(f"{modality_types[i]}-other Loss\tother-{modality_types[i]} Loss\t" for i in range(len(modality_types))))
            f.write("".join(f"{modality_types[i]}-other Accuracy\tother-{modality_types[i]} Accuracy\t" for i in range(len(modality_types))))

        f.write("Temperature\n")
        f.flush()

        for epoch in range(epoch_resume, epochs):
            for split in (["pretrain", "valid"] if epoch != -1 else ["valid"]):
                logger.info(f"Epoch: {epoch}; Split: {split}")
                # 使用pin_memory=True和persistent_workers=True优化数据加载
                dataloader = torch.utils.data.DataLoader(
                    dataset[split],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    drop_last=(split == "pretrain"),
                    pin_memory=True,
                    persistent_workers=True if num_workers > 0 else False)

                for model_key, model in model_dict.items():
                    model.train(split == "pretrain")

                if mode == "pairwise":
                    total_loss = 0.
                    total_pairwise_loss = np.zeros((num_targets, num_targets), dtype=float)
                    total_correct = np.zeros((num_targets, num_targets), dtype=int)
                    total_n = 0
                    total_pairs = np.zeros((num_targets, num_targets), dtype=int)
                elif mode == "leave_one_out":
                    total_loss = 0.
                    total_pairwise_loss = np.zeros((num_targets, 2), dtype=float)
                    total_correct = np.zeros((num_targets, 2), dtype=int)
                    total_n = 0
                    total_pairs = np.zeros((num_targets, 2), dtype=int)

                count = 0
                with torch.set_grad_enabled(split == "pretrain"):
                    with tqdm.tqdm(total=len(dataloader)) as pbar:
                        for data in dataloader:
                            if num_targets == 3:
                                resp, sleep, ekg = data
                                resp = resp.to(device, dtype=torch.float)
                                sleep = sleep.to(device, dtype=torch.float)
                                ekg = ekg.to(device, dtype=torch.float)

                                emb = [
                                    model_dict["respiratory"](resp),
                                    model_dict["sleep_stages"](sleep),
                                    model_dict["ekg"](ekg),
                                ]
                            elif "respiratory" in modality_types and "sleep_stages" in modality_types:
                                resp, sleep = data
                                resp = resp.to(device, dtype=torch.float)
                                sleep = sleep.to(device, dtype=torch.float)

                                emb = [
                                    model_dict["respiratory"](resp),
                                    model_dict["sleep_stages"](sleep),
                                ]
                            elif "respiratory" in modality_types and "ekg" in modality_types:
                                resp, ekg = data
                                resp = resp.to(device, dtype=torch.float)
                                ekg = ekg.to(device, dtype=torch.float)

                                emb = [
                                    model_dict["respiratory"](resp),
                                    model_dict["ekg"](ekg),
                                ]
                            elif "sleep_stages" in modality_types and "ekg" in modality_types:
                                sleep, ekg = data
                                sleep = sleep.to(device, dtype=torch.float)
                                ekg = ekg.to(device, dtype=torch.float)

                                emb = [
                                    model_dict["sleep_stages"](sleep),
                                    model_dict["ekg"](ekg),
                                ]

                            for i in range(num_targets):
                                emb[i] = torch.nn.functional.normalize(emb[i])

                            if mode == "pairwise":
                                loss = 0.
                                pairwise_loss = np.zeros((num_targets, num_targets), dtype=float)
                                correct = np.zeros((num_targets, num_targets), dtype=int)
                                pairs = np.zeros((num_targets, num_targets), dtype=int)

                                for i in range(num_targets):
                                    for j in range(i + 1, num_targets):
                                        # 使用标准SimCLR写法计算logits
                                        similarity = torch.matmul(emb[i], emb[j].transpose(0, 1))
                                        logits = similarity / temperature.clamp(min=min_temp, max=max_temp)
                                        labels = torch.arange(logits.shape[0], device=device)

                                        l = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                                        loss += l
                                        pairwise_loss[i, j] = l.item()
                                        if len(logits) != 0:
                                            correct[i, j] = (torch.argmax(logits, axis=0) == labels).sum().item()
                                        else:
                                            correct[i, j] = 0
                                        pairs[i, j] = batch_size

                                        l = torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels, reduction="sum")
                                        loss += l
                                        pairwise_loss[j, i] = l.item()
                                        if len(logits) != 0:
                                            correct[j, i] = (torch.argmax(logits, axis=1) == labels).sum().item()
                                        else:
                                            correct[j, i] = 0
                                        pairs[j, i] = batch_size
                                loss /= len(ij)
                            if mode == "leave_one_out":
                                loss = 0.
                                pairwise_loss = np.zeros((num_targets, 2), dtype=float)
                                correct = np.zeros((num_targets, 2), dtype=int)
                                pairs = np.zeros((num_targets, 2), dtype=int)

                                # 找出呼吸信号的索引（如果存在）
                                resp_idx = -1
                                if "respiratory" in modality_types:
                                    resp_idx = modality_types.index("respiratory")

                                for i in range(num_targets):
                                    # 为呼吸模态赋予更高的权重
                                    if i == resp_idx:  # 当前模态是呼吸信号
                                        # 其他模态的表示保持不变
                                        other_indices = [j for j in range(num_targets) if j != i]
                                        other_emb = torch.stack([emb[j] for j in other_indices]).sum(0) / (num_targets - 1)
                                    else:  # 当前模态不是呼吸信号
                                        # 如果存在呼吸信号，给它更高的权重
                                        if resp_idx >= 0:
                                            # 计算加权平均，增加呼吸信号的权重
                                            other_indices = [j for j in range(num_targets) if j != i]

                                            # 创建权重列表
                                            weights = []

                                            # 动态调整呼吸信号权重
                                            current_resp_weight = respiratory_weight

                                            # 如果启用动态权重调整，根据模态准确率调整权重
                                            if dynamic_weight and split == "pretrain" and count > 0:
                                                # 获取呼吸信号的准确率
                                                if mode == "leave_one_out" and resp_idx < num_targets:
                                                    resp_acc = 0
                                                    if (total_pairs[resp_idx, 0] + total_pairs[resp_idx, 1]) > 0:
                                                        resp_acc = (total_correct[resp_idx, 0] + total_correct[resp_idx, 1]) / (total_pairs[resp_idx, 0] + total_pairs[resp_idx, 1])

                                                    # 获取其他模态的平均准确率
                                                    other_accs = []
                                                    for k in range(num_targets):
                                                        if k != resp_idx and (total_pairs[k, 0] + total_pairs[k, 1]) > 0:
                                                            acc = (total_correct[k, 0] + total_correct[k, 1]) / (total_pairs[k, 0] + total_pairs[k, 1])
                                                            other_accs.append(acc)

                                                    other_avg_acc = np.mean(other_accs) if other_accs else 0

                                                    # 如果呼吸信号准确率低于其他模态，增加其权重
                                                    if resp_acc < other_avg_acc * 0.9:
                                                        current_resp_weight = min(respiratory_weight * 1.1, 2.0)
                                                    # 如果呼吸信号准确率高于其他模态，减少其权重
                                                    elif resp_acc > other_avg_acc * 1.1:
                                                        current_resp_weight = max(respiratory_weight * 0.95, 1.0)

                                            for j in other_indices:
                                                if j == resp_idx:
                                                    weights.append(current_resp_weight)  # 呼吸信号权重更高
                                                else:
                                                    weights.append(1.0)  # 其他模态标准权重

                                            # 归一化权重
                                            weights = [w / sum(weights) for w in weights]

                                            # 使用权重计算加权平均
                                            other_emb = torch.zeros_like(emb[0])
                                            for idx, j in enumerate(other_indices):
                                                other_emb += weights[idx] * emb[j]
                                        else:
                                            # 没有呼吸信号，使用普通平均
                                            other_indices = [j for j in range(num_targets) if j != i]
                                            other_emb = torch.stack([emb[j] for j in other_indices]).sum(0) / (num_targets - 1)

                                    # 计算与其他模态的对比损失
                                    # 使用标准SimCLR写法计算logits
                                    similarity = torch.matmul(emb[i], other_emb.transpose(0, 1))
                                    logits = similarity / temperature.clamp(min=min_temp, max=max_temp)
                                    labels = torch.arange(logits.shape[0], device=device)

                                    l = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
                                    loss += l
                                    pairwise_loss[i, 0] = l.item()
                                    if len(logits) != 0:
                                        correct[i, 0] = (torch.argmax(logits, axis=0) == labels).sum().item()
                                    else:
                                        correct[i, 0] = 0
                                    pairs[i, 0] = batch_size

                                    l = torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels, reduction="sum")
                                    loss += l
                                    pairwise_loss[i, 1] = l.item()
                                    if len(logits) != 0:
                                        correct[i, 1] = (torch.argmax(logits, axis=1) == labels).sum().item()
                                    else:
                                        correct[i, 1] = 0
                                    pairs[i, 1] = batch_size
                                loss /= num_targets * 2

                            total_loss += loss.item()
                            total_pairwise_loss += pairwise_loss
                            total_correct += correct
                            total_n += batch_size  # 使用实际批次大小
                            total_pairs += pairs

                            # 计算每批次的平均损失
                            batch_loss = loss / batch_size

                            # 不在每个批次输出对比损失，改为在epoch结束后输出

                            # 实现梯度累积
                            if split == "pretrain":
                                # 将损失除以梯度累积步数，相当于平均多个批次的梯度
                                (loss / gradient_accumulation_steps).backward()

                                # 每 gradient_accumulation_steps 步更新一次参数
                                if (count + 1) % gradient_accumulation_steps == 0:
                                    optim.step()
                                    optim.zero_grad()

                                    # 在每次梯度更新后清理内存
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()

                            # 自适应温度调整
                            if adaptive_temp and split == "pretrain":
                                # 根据当前批次准确率自适应调整温度
                                if mode == "pairwise":
                                    # 计算所有模态对之间的平均准确率
                                    accuracies = []
                                    for i in range(num_targets):
                                        for j in range(i + 1, num_targets):
                                            if total_pairs[i, j] > 0:
                                                acc = (total_correct[i, j] + total_correct[j, i]) / (2 * total_pairs[i, j])
                                                accuracies.append(acc)
                                    accuracy = np.mean(accuracies) if accuracies else 0
                                else:  # leave_one_out
                                    # 计算所有模态的平均准确率
                                    accuracies = []
                                    for i in range(num_targets):
                                        if (total_pairs[i, 0] + total_pairs[i, 1]) > 0:
                                            acc = (total_correct[i, 0] + total_correct[i, 1]) / (total_pairs[i, 0] + total_pairs[i, 1])
                                            accuracies.append(acc)
                                    accuracy = np.mean(accuracies) if accuracies else 0

                                    # 不在每个批次记录模态准确率，改为在epoch结束后输出

                                # 根据准确率动态调整温度（对于除法形式，较小的温度使任务更困难）
                                with torch.no_grad():
                                    # 使用更激进的温度调整策略，特别是在训练早期
                                    current_epoch_progress = epoch / epochs  # 训练进度（0-1）

                                    # 根据训练进度调整温度变化率
                                    # 训练早期使用更大的调整幅度，后期逐渐减小
                                    adjust_rate = temp_adjust_factor * (1.0 - current_epoch_progress)

                                    if accuracy > 0.85:
                                        # 准确率高，降低温度使任务更困难
                                        new_temp = temperature * (1.0 - adjust_rate)
                                        temperature.fill_(new_temp.item())
                                    elif accuracy < 0.65:
                                        # 准确率低，增加温度使任务更容易
                                        new_temp = temperature * (1.0 + adjust_rate)
                                        temperature.fill_(new_temp.item())

                            # 确保温度参数在合理范围内
                            with torch.no_grad():
                                if temperature < min_temp:
                                    temperature.fill_(min_temp)  # 设置最小值，避免除以0
                                elif temperature > max_temp:  # 设置一个合理的最大值
                                    temperature.fill_(max_temp)

                            # 显示总体损失、准确率和温度，但不显示模态间损失
                            if mode == "pairwise":
                                # 计算各模态对的准确率
                                acc_info = " ".join(map("{:.1f}".format, [100 * (total_correct[i, j] + total_correct[j, i]) / 2 / total_pairs[i, j] if total_pairs[i, j] > 0 else 0 for i in range(len(modality_types)) for j in range(i + 1, len(modality_types))]))

                                pbar.set_postfix_str(
                                    f"Loss: {total_loss / total_n:.5f} ({batch_loss:.5f}); " +
                                    f"Acc: {acc_info}; " +
                                    f"Temp: {temperature.item():.3f}"
                                )
                            elif mode == "leave_one_out":
                                # 计算各模态的准确率
                                acc_info = " ".join(map("{:.1f}".format, [100 * (total_correct[i, 0] + total_correct[i, 1]) / (total_pairs[i, 0] + total_pairs[i, 1]) if (total_pairs[i, 0] + total_pairs[i, 1]) > 0 else 0 for i in range(len(modality_types))]))

                                pbar.set_postfix_str(
                                    f"Loss: {total_loss / total_n:.5f} ({batch_loss:.5f}); " +
                                    f"Acc: {acc_info}; " +
                                    f"Temp: {temperature.item():.3f}"
                                )
                            pbar.update()
                            count += 1

                if mode == "pairwise":
                    f.write("{}\t{}\t".format(epoch, split))
                    f.write(((len(ij) + 1) * "{:.5f}\t").format(total_loss / total_n, *[total_pairwise_loss[i, j] / total_pairs[i, j] if total_pairs[i, j] > 0 else 0 for (i, j) in ij]))
                    f.write((len(ij) * "{:.3f}\t").format(*[100 * total_correct[i, j] / total_pairs[i, j] if total_pairs[i, j] > 0 else 0 for (i, j) in ij]))
                    f.write("{:.5f}\n".format(temperature.item()))
                elif mode == "leave_one_out":
                    f.write("{}\t{}\t".format(epoch, split))
                    f.write(((num_targets  * 2 + 1) * "{:.5f}\t").format(total_loss / total_n, *[total_pairwise_loss[i, j] / total_pairs[i, j] if total_pairs[i, j] > 0 else 0 for i in range(num_targets) for j in [0, 1]]))
                    f.write(((num_targets  * 2) * "{:.3f}\t").format(*[100 * total_correct[i, j] / total_pairs[i, j] if total_pairs[i, j] > 0 else 0 for i in range(num_targets) for j in [0, 1]]))
                    f.write("{:.5f}\n".format(temperature.item()))
                f.flush()

            # 使用验证集损失更新学习率调度器
            scheduler.step(total_loss / total_n)

            # 更彻底地清理内存
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 强制进行垃圾回收
            import gc
            gc.collect()

            # 在验证集结束后，确保临时变量被释放，但保留我们后续需要的变量
            if split == "valid":
                # 保存我们后续需要的变量
                saved_total_loss = total_loss
                saved_total_pairwise_loss = total_pairwise_loss.copy()
                saved_total_correct = total_correct.copy()
                saved_total_n = total_n
                saved_total_pairs = total_pairs.copy()

                # 释放不需要的临时变量
                if 'emb' in locals():
                    del emb
                if 'loss' in locals():
                    del loss
                if 'pairwise_loss' in locals():
                    del pairwise_loss
                if 'correct' in locals():
                    del correct
                if 'pairs' in locals():
                    del pairs

                # 再次强制垃圾回收和清理GPU缓存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 恢复我们需要的变量
                total_loss = saved_total_loss
                total_pairwise_loss = saved_total_pairwise_loss
                total_correct = saved_total_correct
                total_n = saved_total_n
                total_pairs = saved_total_pairs

            loss = total_loss / total_n
            is_best = (loss < best_loss)
            if is_best:
                best_loss = loss

            save = {
                "epoch": epoch,
                "temperature": temperature.item(),
                "optim_dict": optim.state_dict(),
                "scheduler_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "loss": loss
            }

            for model_key, model in model_dict.items():
                save[f"{model_key}_state_dict"] = model.state_dict()

            if is_best:
                torch.save(save, os.path.join(output, ".best.pt"))
                shutil.move(
                    os.path.join(output, ".best.pt"),
                    os.path.join(output, "best.pt")
                )
            torch.save(save, os.path.join(output, ".checkpoint.pt"))
            shutil.move(
                os.path.join(output, ".checkpoint.pt"),
                os.path.join(output, "checkpoint.pt")
            )

            # 计算并记录每个模态的平均准确率和损失
            modality_accuracies = {}
            modality_losses = {}

            # 在每个epoch结束后输出各模态之间的对比损失
            logger.info(f"Epoch {epoch} 各模态对比损失:")

            # 在输出详细信息前先清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

            if mode == "pairwise":
                # 输出每对模态之间的对比损失
                for i in range(num_targets):
                    for j in range(i + 1, num_targets):
                        mod_i = modality_types[i]
                        mod_j = modality_types[j]
                        if total_pairs[i, j] > 0 and total_pairs[j, i] > 0:
                            loss_ij = total_pairwise_loss[i, j] / total_pairs[i, j]
                            loss_ji = total_pairwise_loss[j, i] / total_pairs[j, i]
                            acc_ij = 100 * total_correct[i, j] / total_pairs[i, j]
                            acc_ji = 100 * total_correct[j, i] / total_pairs[j, i]
                            logger.info(f"  {mod_i}-{mod_j}: 损失 {loss_ij:.5f}, 准确率 {acc_ij:.2f}%")
                            logger.info(f"  {mod_j}-{mod_i}: 损失 {loss_ji:.5f}, 准确率 {acc_ji:.2f}%")

            elif mode == "leave_one_out":
                # 输出每个模态与其他模态组合的对比损失
                for i in range(num_targets):
                    mod_i = modality_types[i]
                    if (total_pairs[i, 0] + total_pairs[i, 1]) > 0:
                        loss_i_other = total_pairwise_loss[i, 0] / total_pairs[i, 0] if total_pairs[i, 0] > 0 else 0
                        loss_other_i = total_pairwise_loss[i, 1] / total_pairs[i, 1] if total_pairs[i, 1] > 0 else 0
                        acc_i_other = 100 * total_correct[i, 0] / total_pairs[i, 0] if total_pairs[i, 0] > 0 else 0
                        acc_other_i = 100 * total_correct[i, 1] / total_pairs[i, 1] if total_pairs[i, 1] > 0 else 0
                        logger.info(f"  {mod_i}-other: 损失 {loss_i_other:.5f}, 准确率 {acc_i_other:.2f}%")
                        logger.info(f"  other-{mod_i}: 损失 {loss_other_i:.5f}, 准确率 {acc_other_i:.2f}%")

            # 计算每个模态的平均性能
            for i, modality in enumerate(modality_types):
                if mode == "leave_one_out":
                    if (total_pairs[i, 0] + total_pairs[i, 1]) > 0:
                        acc = (total_correct[i, 0] + total_correct[i, 1]) / (total_pairs[i, 0] + total_pairs[i, 1])
                        modality_accuracies[modality] = acc * 100  # 转换为百分比
                        # 计算该模态的平均损失
                        mod_loss = (total_pairwise_loss[i, 0] + total_pairwise_loss[i, 1]) / (total_pairs[i, 0] + total_pairs[i, 1])
                        modality_losses[modality] = mod_loss
                elif mode == "pairwise":
                    # 计算每个模态与其他模态的平均准确率和损失
                    acc_sum = 0
                    loss_sum = 0
                    pair_count = 0
                    for j in range(num_targets):
                        if i != j and total_pairs[i, j] > 0:
                            acc_sum += total_correct[i, j] / total_pairs[i, j]
                            loss_sum += total_pairwise_loss[i, j] / total_pairs[i, j]
                            pair_count += 1
                    if pair_count > 0:
                        modality_accuracies[modality] = (acc_sum / pair_count) * 100
                        modality_losses[modality] = loss_sum / pair_count

            # 记录详细的训练信息
            logger.info(f"Epoch {epoch} Summary:")
            logger.info(f"  Loss: {loss:.5f} (Best: {best_loss:.5f})")
            logger.info(f"  Temperature: {temperature.item():.5f}")

            # 输出各模态的平均准确率和损失
            logger.info("  各模态平均性能:")
            for modality in modality_types:
                acc = modality_accuracies.get(modality, 0)
                mod_loss = modality_losses.get(modality, 0)
                logger.info(f"    {modality}: 准确率 {acc:.2f}%, 损失 {mod_loss:.5f}")

            logger.info(f"Saved models. Best loss: {best_loss:.5f}, Current loss: {loss:.5f}")


def count_parameters(model):
    total_params = 0
    total_layers = 0

    for _, param in model.named_parameters():
        total_params += param.numel()
        total_layers += 1

    return total_layers, total_params

if __name__ == '__main__':
    # 配置日志
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])

    pretrain()