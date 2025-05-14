#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to analyze the content of a .arousal file from the PhysioNet 2018 Challenge dataset.
This script reads and displays information about sleep stages and respiratory events.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_and_install_dependencies():
    """检查并安装所需的依赖库。"""
    try:
        import wfdb
        logger.info(f"WFDB库已安装 (版本: {wfdb.__version__})")
    except ImportError:
        logger.info("未找到WFDB库，正在安装...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wfdb"])
        import wfdb
        logger.info(f"WFDB库已安装 (版本: {wfdb.__version__})")
    return True

def load_record_info(record_path):
    """
    加载记录信息（采样率、信号长度等）

    参数:
        record_path: 记录路径（不含扩展名）

    返回:
        dict: 记录信息字典
    """
    import wfdb

    try:
        # 尝试读取头文件
        record = wfdb.rdheader(record_path)
        fs = record.fs  # 采样率
        n_samples = record.sig_len  # 样本数
        n_signals = record.n_sig  # 信号数量
        signal_names = record.sig_name  # 信号名称

        return {
            "fs": fs,
            "n_samples": n_samples,
            "n_signals": n_signals,
            "signal_names": signal_names,
            "record_name": os.path.basename(record_path),
            "duration_seconds": n_samples / fs,
            "duration_hours": n_samples / fs / 3600
        }
    except Exception as e:
        logger.warning(f"无法读取记录 {record_path} 的头文件: {str(e)}")
        # 使用默认值
        return {
            "fs": 200,
            "n_samples": None,
            "record_name": os.path.basename(record_path),
            "error": str(e)
        }

def extract_events_from_arousal_file(arousal_file):
    """
    从arousal文件中提取事件

    参数:
        arousal_file: arousal文件路径

    返回:
        dict: 包含事件信息的字典
    """
    import wfdb

    try:
        # 获取记录路径（去掉.arousal扩展名）
        record_path = arousal_file.rsplit('.', 1)[0]
        record_info = load_record_info(record_path)

        # 读取WFDB注释文件
        annotation = wfdb.rdann(record_path, 'arousal')

        # 提取基本注释信息
        samples = annotation.sample  # 样本点
        symbols = annotation.symbol  # 注释符号
        aux_notes = annotation.aux_note if hasattr(annotation, 'aux_note') else []

        logger.info(f"记录 {record_info['record_name']} - 注释样本数: {len(samples)}")
        logger.info(f"唯一符号: {set(symbols)}")
        if aux_notes:
            logger.info(f"唯一辅助注释: {set(aux_notes)}")

        # 提取睡眠阶段
        sleep_stages = []
        sleep_stage_map = {
            'W': 'Wake',
            'N1': 'N1',
            'N2': 'N2',
            'N3': 'N3',
            'R': 'REM'
        }

        # 提取呼吸事件
        respiratory_events = []
        apnea_keywords = ['apnea', 'apnoea']
        hypopnea_keywords = ['hypopnea', 'hypopoea']

        # 事件的开始和结束配对
        event_start = None
        event_type = None

        # 用于跟踪事件开始的字典
        event_starts = {}

        # 处理所有注释
        for i in range(len(samples)):
            sample = samples[i]
            symbol = symbols[i] if i < len(symbols) else ""
            aux_note = aux_notes[i] if i < len(aux_notes) else ""

            # 检查睡眠阶段注释 - 在PhysioNet 2018数据集中，睡眠阶段存储在aux_note中
            if aux_note in ['W', 'N1', 'N2', 'N3', 'R']:
                # 这是一个睡眠阶段注释
                sleep_stages.append({
                    'sample': sample,
                    'time_seconds': sample / record_info['fs'],
                    'stage': sleep_stage_map.get(aux_note, aux_note)
                })

            # 检查呼吸事件开始 - 格式为 "(resp_hypopnea" 或 "(resp_centralapnea" 等
            if aux_note and '(resp_' in aux_note:
                event_type = None
                if 'hypopnea' in aux_note.lower():
                    event_type = "hypopnea"
                elif 'apnea' in aux_note.lower() or 'apnoea' in aux_note.lower():
                    event_type = "apnea"

                if event_type:
                    # 记录事件开始
                    event_starts[aux_note.replace('(', '')] = {
                        'sample': sample,
                        'type': event_type
                    }

            # 检查呼吸事件结束 - 格式为 "resp_hypopnea)" 或 "resp_centralapnea)" 等
            if aux_note and aux_note.endswith(')') and 'resp_' in aux_note:
                # 提取事件类型（去掉结尾的")"）
                event_type_key = aux_note.replace(')', '')

                # 查找对应的开始事件
                if event_type_key in event_starts:
                    start_info = event_starts[event_type_key]
                    event_start = start_info['sample']
                    event_type = start_info['type']
                    event_end = sample
                    duration = event_end - event_start

                    # 添加到呼吸事件列表
                    respiratory_events.append({
                        'start_sample': event_start,
                        'end_sample': event_end,
                        'start_time': event_start / record_info['fs'],
                        'end_time': event_end / record_info['fs'],
                        'duration_seconds': duration / record_info['fs'],
                        'type': event_type,
                        'description': aux_note
                    })

                    # 从字典中移除已处理的事件
                    del event_starts[event_type_key]

        # 计算统计信息
        apnea_events = [e for e in respiratory_events if e['type'] == 'apnea']
        hypopnea_events = [e for e in respiratory_events if e['type'] == 'hypopnea']

        # 计算AHI（呼吸暂停低通气指数）
        total_hours = record_info['duration_hours'] if 'duration_hours' in record_info else 0
        ahi = (len(apnea_events) + len(hypopnea_events)) / total_hours if total_hours > 0 else 0

        return {
            'record_info': record_info,
            'annotation_info': {
                'total_annotations': len(samples),
                'unique_symbols': list(set(symbols)),
                'unique_aux_notes': list(set(aux_notes)) if aux_notes else []
            },
            'sleep_stages': sleep_stages,
            'respiratory_events': respiratory_events,
            'statistics': {
                'total_sleep_stages': len(sleep_stages),
                'total_respiratory_events': len(respiratory_events),
                'apnea_count': len(apnea_events),
                'hypopnea_count': len(hypopnea_events),
                'total_hours': total_hours,
                'ahi': ahi
            }
        }

    except Exception as e:
        logger.error(f"处理文件 {arousal_file} 时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': str(e), 'traceback': traceback.format_exc()}

def visualize_events(data, output_file=None):
    """
    Visualize sleep stages and respiratory events

    Args:
        data: Data dictionary from extract_events_from_arousal_file
        output_file: Path to save the visualization (optional)
    """
    if 'error' in data:
        logger.error(f"Cannot visualize data with errors: {data['error']}")
        return

    # Use default font settings to avoid Chinese font issues
    try:
        # Reset to default parameters
        plt.rcParams.update(plt.rcParamsDefault)
        # Set basic plotting parameters
        plt.rcParams['figure.figsize'] = [15, 10]
        plt.rcParams['figure.dpi'] = 100
        # Ensure we're using English locale
        import locale
        locale.setlocale(locale.LC_ALL, 'C')
    except Exception as e:
        logger.warning(f"Error setting font: {str(e)}")

    record_info = data['record_info']
    sleep_stages = data['sleep_stages']
    respiratory_events = data['respiratory_events']

    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    # Set figure title with English only
    plt.suptitle(f'Analysis of {record_info["record_name"]}', fontsize=16)

    # 计算X轴范围 - 使用记录的总时长
    total_hours = record_info.get('duration_hours', 8)  # 默认8小时
    x_min = 0
    x_max = total_hours

    # 创建共享X轴的子图
    ax1 = plt.subplot(2, 1, 1)  # 睡眠阶段图
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)  # 呼吸事件图，共享X轴

    # 绘制睡眠阶段
    if sleep_stages:
        # 转换为DataFrame以便于绘图
        stages_df = pd.DataFrame(sleep_stages)

        # 将睡眠阶段映射为数值以便于绘图
        stage_to_num = {
            'Wake': 5,
            'N1': 4,
            'N2': 3,
            'N3': 2,
            'REM': 1
        }

        stages_df['stage_num'] = stages_df['stage'].map(stage_to_num)

        # 绘制睡眠阶段 - 使用英文标签
        ax1.step(stages_df['time_seconds'] / 3600, stages_df['stage_num'], where='post')
        ax1.set_yticks(list(stage_to_num.values()))
        ax1.set_yticklabels(list(stage_to_num.keys()))
        ax1.set_ylabel('Sleep Stage')
        ax1.set_title(f'Sleep Stages for {record_info["record_name"]}')
        ax1.grid(True)

        # 设置X轴范围
        ax1.set_xlim(x_min, x_max)

    # 绘制呼吸事件
    if respiratory_events:
        # 分离呼吸暂停和低通气事件
        apnea_events = [e for e in respiratory_events if e['type'] == 'apnea']
        hypopnea_events = [e for e in respiratory_events if e['type'] == 'hypopnea']

        # 将事件绘制为点而不是线，使图更紧凑
        for event in apnea_events:
            # 计算事件中点
            mid_time = (event['start_time'] + event['end_time']) / 2 / 3600
            ax2.plot(mid_time, 1, 'ro', markersize=4)  # 红色圆点表示呼吸暂停

        for event in hypopnea_events:
            # 计算事件中点
            mid_time = (event['start_time'] + event['end_time']) / 2 / 3600
            ax2.plot(mid_time, 0, 'bo', markersize=4)  # 蓝色圆点表示低通气

        # 设置Y轴更紧凑 - 使用英文标签
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Hypopnea', 'Apnea'])
        ax2.set_ylim(-0.2, 1.2)  # 减小Y轴范围，使图更紧凑
        ax2.set_xlabel('Time (hours)')
        ax2.set_title(f'Respiratory Events for {record_info["record_name"]}')
        ax2.grid(True)

        # 设置X轴范围
        ax2.set_xlim(x_min, x_max)

    # 确保X轴刻度一致
    plt.setp(ax1.get_xticklabels(), visible=False)  # 隐藏上图的X轴标签
    ax2.set_xlabel('Time (hours)')

    # 调整布局，留出顶部空间给标题
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_file:
        plt.savefig(output_file, dpi=300)  # 增加DPI以提高清晰度
        logger.info(f"可视化图表已保存到 {output_file}")
    else:
        plt.show()

def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description='分析PhysioNet 2018挑战赛数据集中的.arousal文件')
    parser.add_argument('arousal_file', help='.arousal文件路径')
    parser.add_argument('--output', '-o', help='可视化输出文件路径（可选）')
    parser.add_argument('--verbose', '-v', action='store_true', help='启用详细输出')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # 检查并安装依赖
    check_and_install_dependencies()

    # 从arousal文件中提取事件
    logger.info(f"正在分析arousal文件: {args.arousal_file}")
    data = extract_events_from_arousal_file(args.arousal_file)

    # 打印统计信息
    if 'error' in data:
        logger.error(f"分析文件时出错: {data['error']}")
        return

    stats = data['statistics']
    logger.info(f"记录: {data['record_info']['record_name']}")
    logger.info(f"持续时间: {stats['total_hours']:.2f} 小时")
    logger.info(f"总注释数: {data['annotation_info']['total_annotations']}")
    logger.info(f"睡眠阶段: {stats['total_sleep_stages']}")
    logger.info(f"呼吸事件: {stats['total_respiratory_events']} (呼吸暂停: {stats['apnea_count']}, 低通气: {stats['hypopnea_count']})")
    logger.info(f"AHI: {stats['ahi']:.2f}")

    # 可视化事件
    if args.output:
        visualize_events(data, args.output)
    else:
        visualize_events(data)

if __name__ == "__main__":
    main()
