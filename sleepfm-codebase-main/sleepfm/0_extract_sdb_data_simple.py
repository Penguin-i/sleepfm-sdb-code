#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版SDB（睡眠呼吸障碍）数据预处理脚本

该脚本从PhysioNet Challenge 2018数据集中提取SDB事件，
并创建用于训练SDB检测模型的数据集。
"""

import numpy as np
import mne
import wfdb
from tqdm import tqdm
import multiprocessing
import logging
import warnings
import pickle
import sys
import os
import glob
import scipy.signal
import argparse
from scipy.io import loadmat

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from sleepfm.config import ALL_CHANNELS, SDB_CONFIG, CHANNEL_DATA
except ImportError:
    # 尝试相对导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import ALL_CHANNELS, SDB_CONFIG, CHANNEL_DATA

# SDB事件类型字典
SDB_EVENT_TYPES = {
    "non_sdb": 0,   # 非SDB事件
    "apnea": 1,     # 呼吸暂停事件
    "hypopnea": 2   # 低通气事件
}

# SDB事件ID到名称的映射
SDB_ID_TO_EVENT = {v: k for k, v in SDB_EVENT_TYPES.items()}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 忽略所有警告
warnings.filterwarnings("ignore")

def apply_channel_wise_zscore(data, epsilon=1e-8):
    """对数据的每个通道应用Z-score标准化"""
    # 创建输出数组
    normalized_data = np.zeros_like(data, dtype=np.float32)

    # 对每个通道进行标准化
    for i in range(data.shape[0]):
        channel_data = data[i, :]
        mean = np.mean(channel_data)
        std = np.std(channel_data)

        # 防止除零错误
        if std == 0:
            std = epsilon

        # 应用标准化
        normalized_data[i, :] = (channel_data - mean) / (std + epsilon)

    return normalized_data

def import_signal_names(file_name):
    """从.hea头文件中提取信号名称、采样率和样本数信息"""
    with open(file_name, 'r') as myfile:
        s = myfile.read()
        s = s.split('\n')
        s = [x.split() for x in s]

        n_signals = int(s[0][1])  # 信号数量
        n_samples = int(s[0][3])  # 样本数
        Fs = int(s[0][2])  # 采样率

        s = s[1:-1]  # 去除头尾
        s = [s[i][8] for i in range(0, n_signals)]
    return s, Fs, n_samples

def extract_sdb_events(arousal_file):
    """从.arousal文件中提取SDB事件（呼吸暂停和低通气）"""
    try:
        # 去掉.arousal后缀
        record_path = arousal_file.replace('.arousal', '')

        # 使用wfdb读取标注
        annotations = wfdb.rdann(record_path, 'arousal')

        # 提取标注信息
        sample_points = annotations.sample  # 样本点
        aux_info = annotations.aux_note  # 辅助信息

        # 初始化SDB事件列表
        sdb_events = []

        # 查找SDB相关事件
        for i, aux in enumerate(aux_info):
            if "resp_" in aux.lower():
                # 检查是否是事件开始
                if "(" in aux:
                    event_type = aux.replace("(", "").strip()
                    start_sample = sample_points[i]

                    # 查找事件结束
                    for j in range(i+1, len(aux_info)):
                        if ")" in aux_info[j] and aux_info[j].replace(")", "").strip() == event_type:
                            end_sample = sample_points[j]
                            duration = end_sample - start_sample

                            # 确定事件类型
                            event_id = SDB_EVENT_TYPES["non_sdb"]

                            # 检查是否是呼吸暂停事件
                            if any(apnea_keyword.lower() in event_type.lower() for apnea_keyword in SDB_CONFIG["event_types"]["apnea"]):
                                event_id = SDB_EVENT_TYPES["apnea"]
                            # 检查是否是低通气事件
                            elif any(hypopnea_keyword.lower() in event_type.lower() for hypopnea_keyword in SDB_CONFIG["event_types"]["hypopnea"]):
                                event_id = SDB_EVENT_TYPES["hypopnea"]

                            # 如果是SDB事件，添加到列表
                            if event_id != SDB_EVENT_TYPES["non_sdb"]:
                                sdb_events.append((start_sample, duration, event_id))
                            break

        # 打印找到的SDB事件数量
        if len(sdb_events) > 0:
            # 统计不同类型的事件数量
            apnea_count = sum(1 for _, _, event_id in sdb_events if event_id == SDB_EVENT_TYPES["apnea"])
            hypopnea_count = sum(1 for _, _, event_id in sdb_events if event_id == SDB_EVENT_TYPES["hypopnea"])
            logger.info(f"找到 {len(sdb_events)} 个SDB事件: {apnea_count} 个呼吸暂停, {hypopnea_count} 个低通气")

        return sdb_events
    except Exception as e:
        logger.error(f"提取SDB事件时出错: {str(e)}")
        return []

def process_sdb_events_to_annotations(sdb_events, fs):
    """将SDB事件转换为MNE注释对象"""
    # 创建注释参数
    onset = [event[0] / fs for event in sdb_events]  # 开始时间（秒）
    duration = [event[1] / fs for event in sdb_events]  # 持续时间（秒）
    description = [SDB_ID_TO_EVENT[event[2]] for event in sdb_events]  # 事件描述

    # 创建MNE注释对象
    annotations = mne.Annotations(onset=onset, duration=duration, description=description)

    return annotations

def get_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="提取SDB数据并保存")
    parser.add_argument("--data_path", type=str,
                        default=SDB_CONFIG["raw_data_path"],
                        help="原始数据路径")
    parser.add_argument("--save_path", type=str,
                        default=SDB_CONFIG["data_path"],
                        help="保存路径")
    parser.add_argument("--num_files", type=int, default=-1,
                        help="处理文件数量，-1表示处理所有文件")
    parser.add_argument("--window_size", type=float, default=SDB_CONFIG["window_size"],
                        help="窗口大小（秒）")
    parser.add_argument("--sdb_threshold", type=float, default=SDB_CONFIG["event_threshold"],
                        help="SDB事件比例阈值，超过此阈值的窗口被视为SDB窗口")
    parser.add_argument("--num_threads", type=int, default=4,
                        help="并行处理线程数")
    parser.add_argument("--target_sampling_rate", type=int, default=SDB_CONFIG["sample_rate"],
                        help="目标采样率")
    parser.add_argument("--apply_filter", action="store_true", default=True,
                        help="是否应用带通滤波，对呼吸信号使用0.1-0.5Hz，对EEG使用0.5-45Hz")
    parser.add_argument("--no_filter", action="store_false", dest="apply_filter",
                        help="不应用带通滤波")
    parser.add_argument("--apply_zscore", action="store_true", default=True,
                        help="是否应用Z-score标准化")
    parser.add_argument("--no_zscore", action="store_false", dest="apply_zscore",
                        help="不应用Z-score标准化")
    parser.add_argument("--balance_ratio", type=float, default=1.0,
                        help="SDB与非SDB窗口的平衡比例，1.0表示保留所有窗口，小于1表示对非SDB窗口进行下采样")

    return parser.parse_args()

def parallel_process_file(args):
    """并行处理文件"""
    file_list = args[0]
    path_to_save = args[1]
    window_size = args[2]
    sdb_threshold = args[3]
    target_sampling_rate = args[4]
    apply_filter = args[5] if len(args) > 5 else True
    apply_zscore = args[6] if len(args) > 6 else True
    balance_ratio = args[7] if len(args) > 7 else 1.0

    # 创建保存目录
    path_to_X = os.path.join(path_to_save, "X")
    path_to_Y = os.path.join(path_to_save, "Y")

    os.makedirs(path_to_X, exist_ok=True)
    os.makedirs(path_to_Y, exist_ok=True)

    # 统计信息
    total_windows = 0
    sdb_windows = 0
    non_sdb_windows = 0
    skipped_windows = 0

    for file_dict in tqdm(file_list):
        # 获取文件路径
        mat_file = file_dict["mat"]
        hea_file = file_dict["hea"]
        arousal_file = file_dict["arousal"]

        # 获取记录名称
        record_name = os.path.basename(mat_file).split('.')[0]

        # 创建患者目录
        path_to_patient_X = os.path.join(path_to_X, record_name)
        path_to_patient_Y = os.path.join(path_to_Y, f"{record_name}.pickle")

        # 如果已处理，则跳过
        if os.path.exists(path_to_patient_X) and len(os.listdir(path_to_patient_X)) > 0:
            logger.info(f"患者已处理: {path_to_patient_X}")
            continue

        try:
            # 步骤1: 导入信号名称和采样率
            signal_names, fs, _ = import_signal_names(hea_file)

            # 步骤2: 使用wfdb读取信号数据
            try:
                record = wfdb.rdrecord(mat_file.replace('.mat', ''))
                data = record.p_signal.T  # 转置以匹配MNE期望的格式
                logger.info(f"使用wfdb读取信号数据成功: {record_name}")
            except Exception as e:
                logger.warning(f"使用wfdb读取失败，尝试使用scipy.io.loadmat: {str(e)}")
                try:
                    # 尝试使用loadmat读取
                    mat_data = loadmat(mat_file)
                    if 'val' in mat_data:
                        data = mat_data['val']
                    else:
                        # 尝试找到主要数据数组
                        for key in mat_data.keys():
                            if isinstance(mat_data[key], np.ndarray) and mat_data[key].size > 100:
                                data = mat_data[key]
                                break
                    logger.info(f"使用scipy.io.loadmat读取信号数据成功: {record_name}")
                except Exception as e2:
                    logger.error(f"读取信号数据失败: {str(e2)}")
                    continue

            # 步骤3: 确保数据形状正确
            if data.shape[0] > data.shape[1]:  # 如果行数大于列数，可能需要转置
                data = data.T

            # 步骤4: 提取SDB事件
            sdb_events = extract_sdb_events(arousal_file)

            if len(sdb_events) == 0:
                logger.warning(f"未找到SDB事件: {record_name}")
                continue

            # 步骤5: 创建MNE信息
            ch_types = ['eeg'] * len(signal_names)  # 假设所有通道都是EEG
            info = mne.create_info(signal_names, fs, ch_types=ch_types)

            # 步骤6: 创建MNE原始数据并应用滤波（如果启用）
            raw = mne.io.RawArray(data, info)

            # 如果启用滤波，应用带通滤波
            if apply_filter:
                # 根据CHANNEL_DATA分组识别不同类型的通道
                respiratory_picks = []  # 呼吸相关通道
                sleep_stage_picks = []  # 睡眠脑电相关通道
                ekg_picks = []          # 心电图通道
                other_picks = []        # 其他通道

                # 通道分组查找表
                channel_group_lookup = {}
                for group_name, channels in CHANNEL_DATA.items():
                    for channel in channels:
                        channel_group_lookup[channel] = group_name

                for i, ch_name in enumerate(raw.ch_names):
                    # 尝试精确匹配
                    if ch_name in channel_group_lookup:
                        group = channel_group_lookup[ch_name]
                        if group == "Respiratory":
                            respiratory_picks.append(i)
                        elif group == "Sleep_Stages":
                            sleep_stage_picks.append(i)
                        elif group == "EKG":
                            ekg_picks.append(i)
                        else:
                            other_picks.append(i)
                    else:
                        # 尝试部分匹配
                        matched = False
                        for channel, group in channel_group_lookup.items():
                            if channel in ch_name or ch_name in channel:
                                if group == "Respiratory":
                                    respiratory_picks.append(i)
                                elif group == "Sleep_Stages":
                                    sleep_stage_picks.append(i)
                                elif group == "EKG":
                                    ekg_picks.append(i)
                                else:
                                    other_picks.append(i)
                                matched = True
                                break
                        if not matched:
                            other_picks.append(i)

                # 对呼吸通道应用0.1-0.5Hz带通滤波
                if respiratory_picks:
                    raw.filter(l_freq=0.1, h_freq=0.5, picks=respiratory_picks,
                              fir_design="firwin", verbose=False)

                # 对睡眠脑电通道应用0.5-45Hz带通滤波
                if sleep_stage_picks:
                    raw.filter(l_freq=0.5, h_freq=45.0, picks=sleep_stage_picks,
                              fir_design="firwin", verbose=False)

                # 对心电图通道应用1-40Hz带通滤波
                if ekg_picks:
                    raw.filter(l_freq=1.0, h_freq=40.0, picks=ekg_picks,
                              fir_design="firwin", verbose=False)

                # 对其他通道应用0.5-45Hz带通滤波（默认处理）
                if other_picks:
                    raw.filter(l_freq=0.5, h_freq=45.0, picks=other_picks,
                              fir_design="firwin", verbose=False)

                logger.info(f"完成信号滤波处理: 呼吸通道 {len(respiratory_picks)}个, " +
                          f"睡眠脑电通道 {len(sleep_stage_picks)}个, " +
                          f"心电通道 {len(ekg_picks)}个, " +
                          f"其他通道 {len(other_picks)}个")

            # 步骤7: 创建SDB注释并设置到原始数据
            annotations = process_sdb_events_to_annotations(sdb_events, fs)
            raw.set_annotations(annotations)

            # 步骤8: 重采样到目标采样率
            raw.resample(sfreq=target_sampling_rate)
            fs_new = target_sampling_rate

            # 步骤9: 创建SDB掩码（用于后续计算每个窗口的SDB比例）
            sdb_mask = np.zeros(len(raw.times))

            # 遍历所有注释，标记SDB事件位置为1
            for annot in raw.annotations:
                onset = annot['onset']  # 开始时间（秒）
                duration = annot['duration']  # 持续时间（秒）
                description = annot['description']  # 事件描述

                # 检查是否是SDB事件
                is_sdb_event = description in ["apnea", "hypopnea"]

                if is_sdb_event:
                    # 转换为样本点
                    start_sample = int(onset * fs_new)
                    duration_samples = int(duration * fs_new)
                    end_sample = min(start_sample + duration_samples, len(raw.times))

                    # 标记SDB事件
                    if start_sample < len(sdb_mask) and end_sample <= len(sdb_mask):
                        sdb_mask[start_sample:end_sample] = 1

            # 步骤10: 创建固定长度的Epochs
            try:
                epochs = mne.make_fixed_length_epochs(
                    raw,
                    duration=window_size,
                    preload=True
                )

                logger.info(f"创建了 {len(epochs)} 个epochs")
            except Exception as e:
                logger.error(f"创建Epochs失败: {record_name}, 错误: {str(e)}")
                continue

            # 创建患者目录
            os.makedirs(path_to_patient_X, exist_ok=True)

            # 保存窗口和标签
            window_labels = {}

            # 记录此记录的窗口统计信息
            record_sdb_windows = 0
            record_non_sdb_windows = 0
            record_total_windows = 0

            # 处理每个Epoch
            for idx, epoch in enumerate(epochs):
                try:
                    # 获取窗口的时间信息
                    # 检查epoch是否为MNE Epoch对象
                    if hasattr(epoch, 'tmin') and hasattr(epoch, 'times'):
                        tmin = epoch.tmin
                        # 检查是否有events属性
                        if hasattr(epoch, 'events') and len(epoch.events) > 0:
                            onset = epoch.events[0, 0] / fs_new + tmin
                        else:
                            # 如果没有events属性，使用idx和window_size计算onset
                            onset = idx * window_size
                        duration = epoch.times[-1] - epoch.times[0]
                    else:
                        # 如果epoch不是MNE Epoch对象，使用默认值
                        onset = idx * window_size
                        duration = window_size

                    # 计算窗口的SDB比例
                    start_sample = int(onset * fs_new)
                    end_sample = int((onset + duration) * fs_new)

                    # 确保索引在有效范围内
                    start_sample = max(0, start_sample)
                    end_sample = min(len(sdb_mask), end_sample)

                    # 提取该窗口的SDB掩码
                    window_mask = sdb_mask[start_sample:end_sample]

                    # 计算SDB事件比例
                    if len(window_mask) > 0:
                        sdb_ratio = np.sum(window_mask) / len(window_mask)
                    else:
                        sdb_ratio = 0

                    # 判断是否为SDB窗口（根据SDB事件比例阈值）
                    is_sdb = 1 if sdb_ratio >= sdb_threshold else 0

                    # 找出与ALL_CHANNELS匹配的通道
                    indices = []
                    for channel in ALL_CHANNELS:
                        if channel in raw.ch_names:
                            indices.append(raw.ch_names.index(channel))
                        else:
                            # 如果找不到精确匹配，尝试部分匹配
                            matched = False
                            for j, name in enumerate(raw.ch_names):
                                if channel in name or name in channel:
                                    indices.append(j)
                                    matched = True
                                    break
                            if not matched:
                                # 如果仍然找不到匹配，使用-1作为占位符
                                indices.append(-1)

                    # 获取数据
                    # 检查epoch是否有get_data方法
                    if hasattr(epoch, 'get_data'):
                        epoch_data = epoch.get_data()[0]  # 获取第一个epoch的数据
                    elif isinstance(epoch, np.ndarray):
                        # 如果epoch是numpy数组，直接使用
                        if len(epoch.shape) == 3:  # [epochs, channels, samples]
                            epoch_data = epoch[0]
                        else:
                            epoch_data = epoch
                    else:
                        # 如果无法获取数据，跳过此窗口
                        logger.warning(f"无法从epoch获取数据，跳过窗口: {record_name}_{idx}")
                        continue

                    # 创建最终窗口数据
                    final_window = np.zeros((len(ALL_CHANNELS), epoch_data.shape[1]))
                    for j, idx_ch in enumerate(indices):
                        if idx_ch >= 0 and idx_ch < epoch_data.shape[0]:
                            # 使用copy()创建连续数组，避免视图问题
                            final_window[j] = np.array(epoch_data[idx_ch]).copy()

                    # 应用Z-score标准化（如果启用）
                    if apply_zscore:
                        final_window = apply_channel_wise_zscore(final_window)

                    # 统计信息
                    total_windows += 1
                    record_total_windows += 1

                    if is_sdb == 1:
                        sdb_windows += 1
                        record_sdb_windows += 1
                    else:
                        non_sdb_windows += 1
                        record_non_sdb_windows += 1

                    # 保存窗口
                    file_name = f"{record_name}_{idx}.npy"
                    np.save(os.path.join(path_to_patient_X, file_name), final_window)

                    # 保存标签
                    window_labels[file_name] = is_sdb

                    # 额外记录窗口的时间信息和SDB比例，便于后续分析
                    window_labels[f"{file_name}_info"] = {
                        "onset": onset,  # 开始时间（秒）
                        "duration": duration,  # 持续时间（秒）
                        "sdb_ratio": sdb_ratio,  # SDB事件比例
                        "label": is_sdb  # 标签
                    }
                except Exception as e:
                    logger.warning(f"处理窗口 {record_name}_{idx} 时出错: {str(e)}，跳过该窗口")
                    skipped_windows += 1
                    continue

            # 保存标签字典
            with open(path_to_patient_Y, 'wb') as f:
                pickle.dump(window_labels, f)

            # 报告此记录的窗口统计信息
            logger.info(f"处理了 {record_total_windows} 个窗口，其中 {record_sdb_windows} 个SDB窗口，{record_non_sdb_windows} 个非SDB窗口")

            if record_total_windows > 0:
                sdb_percent = record_sdb_windows / record_total_windows * 100
                logger.info(f"处理完成: {record_name}, 窗口数: {record_total_windows}, " +
                         f"SDB窗口数: {record_sdb_windows} ({sdb_percent:.2f}%), " +
                         f"非SDB窗口数: {record_non_sdb_windows} ({100-sdb_percent:.2f}%)")

        except Exception as e:
            logger.error(f"处理文件时出错: {record_name}, 错误: {str(e)}")
            continue

    logger.info(f"批处理完成: 总窗口数: {total_windows}, " +
               f"SDB窗口数: {sdb_windows} ({sdb_windows/total_windows*100 if total_windows > 0 else 0:.2f}%), " +
               f"非SDB窗口数: {non_sdb_windows} ({non_sdb_windows/total_windows*100 if total_windows > 0 else 0:.2f}%), " +
               f"跳过窗口数: {skipped_windows}")
    return [1]

def main():
    """主函数"""
    # 获取命令行参数
    args = get_arguments()

    # 设置路径
    path_to_data = args.data_path
    path_to_save = args.save_path

    # 检查路径
    if not os.path.exists(path_to_data):
        logger.error(f"数据路径不存在: {path_to_data}")
        return

    logger.info(f"数据路径: {path_to_data}")
    logger.info(f"保存路径: {path_to_save}")

    # 获取其他参数
    num_files = args.num_files
    window_size = args.window_size
    sdb_threshold = args.sdb_threshold
    num_threads = args.num_threads
    target_sampling_rate = args.target_sampling_rate
    apply_filter = args.apply_filter
    apply_zscore = args.apply_zscore
    balance_ratio = args.balance_ratio

    logger.info(f"处理文件数: {num_files if num_files != -1 else '全部'}")
    logger.info(f"窗口大小: {window_size}秒")
    logger.info(f"SDB阈值: {sdb_threshold}")
    logger.info(f"线程数: {num_threads}")
    logger.info(f"目标采样率: {target_sampling_rate}Hz")
    logger.info(f"应用滤波: {apply_filter}")
    logger.info(f"应用Z-score标准化: {apply_zscore}")
    logger.info(f"SDB与非SDB窗口的平衡比例: {balance_ratio}")

    # 创建保存目录
    os.makedirs(path_to_save, exist_ok=True)

    # 获取所有受试者
    subjects = os.listdir(path_to_data)
    subjects_path = [os.path.join(path_to_data, subject) for subject in subjects]

    # 筛选有效文件
    valid_files = []

    for subject_path in subjects_path:
        # 跳过非目录
        if not os.path.isdir(subject_path) or "RECORDS" in subject_path or "ANNOTATORS" in subject_path:
            continue

        # 获取所有文件
        files = glob.glob(f'{subject_path}/*')

        try:
            temp_dict = {}

            # 检查必要文件
            mat_file = False
            hea_file = False
            arousal_file = False

            for file in files:
                if '.hea' in file:
                    temp_dict["hea"] = file
                    hea_file = True
                elif '.arousal' in file:
                    temp_dict["arousal"] = file
                    arousal_file = True
                elif '.mat' in file and '-arousal.mat' not in file:
                    temp_dict["mat"] = file
                    mat_file = True
                elif '-arousal.mat' in file:
                    temp_dict["arousal_mat"] = file

            # 检查文件是否存在且可读
            all_files_valid = True
            for key, file_path in temp_dict.items():
                if not os.path.exists(file_path) or not os.access(file_path, os.R_OK):
                    logger.warning(f"文件不存在或不可读: {file_path}")
                    all_files_valid = False
                    break

            # 只有当所有必要文件都存在且可读时才添加
            if mat_file and hea_file and arousal_file and all_files_valid:
                valid_files.append(temp_dict)
                logger.info(f"找到有效数据: {subject_path}")
            else:
                missing_files = []
                if not mat_file:
                    missing_files.append(".mat")
                if not hea_file:
                    missing_files.append(".hea")
                if not arousal_file:
                    missing_files.append(".arousal")
                if not all_files_valid:
                    missing_files.append("(文件不可读)")

                logger.warning(f"缺少必要文件: {subject_path}, 缺失: {', '.join(missing_files)}")
        except Exception as e:
            logger.error(f"处理目录时出错: {subject_path}, 错误: {str(e)}")
            continue

    logger.info(f"开始提取SDB数据...")

    # 限制文件数量
    if num_files != -1:
        valid_files = valid_files[:num_files]

    # 划分任务
    files_per_thread = np.array_split(valid_files, num_threads)

    # 创建任务
    tasks = [(file_list, path_to_save, window_size, sdb_threshold, target_sampling_rate,
              apply_filter, apply_zscore, balance_ratio)
             for file_list in files_per_thread]

    # 并行处理
    with multiprocessing.Pool(num_threads) as pool:
        _ = [y for x in pool.imap_unordered(parallel_process_file, tasks) for y in x]

    # 统计结果
    # 计算总窗口数和SDB窗口数
    total_windows = 0
    sdb_windows = 0
    non_sdb_windows = 0
    skipped_windows = 0

    # 遍历所有标签文件
    Y_dir = os.path.join(path_to_save, "Y")
    for label_file in glob.glob(os.path.join(Y_dir, "*.pickle")):
        try:
            with open(label_file, 'rb') as f:
                labels = pickle.load(f)

            # 统计窗口数、SDB窗口数和非SDB窗口数
            window_count = sum(1 for k in labels.keys() if not k.endswith("_info"))
            sdb_count = sum(1 for k, v in labels.items() if not k.endswith("_info") and v == 1)
            non_sdb_count = sum(1 for k, v in labels.items() if not k.endswith("_info") and v == 0)

            # 记录每个文件的窗口信息
            record_name = os.path.basename(label_file).replace('.pickle', '')
            if window_count > 0:
                sdb_percent = sdb_count / window_count * 100
                logger.info(f"文件统计: {record_name}, 窗口数: {window_count}, " +
                          f"SDB: {sdb_count} ({sdb_percent:.2f}%), " +
                          f"非SDB: {non_sdb_count} ({100-sdb_percent:.2f}%)")

            total_windows += window_count
            sdb_windows += sdb_count
            non_sdb_windows += non_sdb_count
        except Exception as e:
            logger.error(f"读取标签文件出错: {label_file}, 错误: {str(e)}")

    logger.info(f"SDB数据提取完成!")

    # 计算全局统计信息
    if total_windows > 0:
        sdb_percent = sdb_windows / total_windows * 100
        non_sdb_percent = non_sdb_windows / total_windows * 100

        logger.info(f"总窗口数: {total_windows}")
        logger.info(f"SDB窗口数: {sdb_windows} ({sdb_percent:.2f}% SDB窗口)")
        logger.info(f"非SDB窗口数: {non_sdb_windows} ({non_sdb_percent:.2f}% 非SDB窗口)")
        logger.info(f"跳过窗口数: {skipped_windows}")
        logger.info(f"SDB/非SDB比例: {sdb_windows}/{non_sdb_windows} = {sdb_windows/non_sdb_windows if non_sdb_windows > 0 else 'inf':.2f}")
    else:
        logger.warning("未处理任何窗口!")

    logger.info(f"数据保存在: {path_to_save}")

    # 打印预处理设置信息
    logger.info(f"预处理设置:")
    logger.info(f"  - 窗口大小: {window_size}秒")
    logger.info(f"  - 目标采样率: {target_sampling_rate}Hz")
    logger.info(f"  - SDB事件阈值: {sdb_threshold}")
    logger.info(f"  - 应用滤波: {apply_filter}")
    logger.info(f"  - 应用Z-score标准化: {apply_zscore}")
    logger.info(f"  - SDB与非SDB窗口的平衡比例: {balance_ratio}")

if __name__ == "__main__":
    main()
