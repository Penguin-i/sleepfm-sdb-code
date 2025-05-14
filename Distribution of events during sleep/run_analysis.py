#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the arousal file analysis on a specific file.
"""

import os
import sys
import argparse
import logging
from analyze_arousal_file import extract_events_from_arousal_file, visualize_events, check_and_install_dependencies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run arousal file analysis')
    parser.add_argument('--file', '-f', default='/root/autodl-tmp/physionet.org/files/challenge-2018/1.0.0/training/tr13-0076/tr13-0076.arousal',
                        help='Path to the .arousal file (default: %(default)s)')
    parser.add_argument('--output-dir', '-d', default='/root/autodl-tmp/processed_data/pc18/sdb',
                        help='Output directory for results (default: %(default)s)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"文件未找到: {args.file}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filenames based on input file
    record_name = os.path.basename(args.file).split('.')[0]
    output_file = os.path.join(args.output_dir, f"{record_name}_analysis.png")
    output_text_file = os.path.join(args.output_dir, f"{record_name}_analysis.txt")

    # 检查并安装依赖
    check_and_install_dependencies()

    # 从arousal文件中提取事件
    logger.info(f"正在分析arousal文件: {args.file}")
    data = extract_events_from_arousal_file(args.file)

    # Print statistics
    if 'error' in data:
        logger.error(f"Error analyzing file: {data['error']}")
        return

    # 打印文件的详细信息
    print("\n" + "="*80)
    print(f"文件分析: {os.path.basename(args.file)}")
    print("="*80)

    # 记录信息
    record_info = data['record_info']
    print("\n记录信息:")
    print(f"记录名称: {record_info['record_name']}")
    print(f"采样率: {record_info['fs']} Hz")
    print(f"样本数量: {record_info['n_samples']}")
    if 'n_signals' in record_info:
        print(f"信号数量: {record_info['n_signals']}")
    if 'signal_names' in record_info:
        print(f"信号名称: {', '.join(record_info['signal_names'])}")
    print(f"持续时间: {record_info['duration_seconds']:.2f} 秒 ({record_info['duration_hours']:.2f} 小时)")

    # 注释信息
    ann_info = data['annotation_info']
    print("\n注释信息:")
    print(f"总注释数: {ann_info['total_annotations']}")
    print(f"唯一符号: {', '.join(ann_info['unique_symbols'])}")
    if ann_info['unique_aux_notes']:
        print(f"唯一辅助注释: {', '.join(ann_info['unique_aux_notes'])}")

    # 睡眠阶段信息
    sleep_stages = data['sleep_stages']
    print("\n睡眠阶段信息:")
    print(f"睡眠阶段注释总数: {len(sleep_stages)}")
    if sleep_stages:
        # 统计每个阶段的出现次数
        stage_counts = {}
        for stage in sleep_stages:
            stage_name = stage['stage']
            if stage_name not in stage_counts:
                stage_counts[stage_name] = 0
            stage_counts[stage_name] += 1

        print("睡眠阶段分布:")
        stage_names_cn = {
            'Wake': '清醒',
            'N1': 'N1期',
            'N2': 'N2期',
            'N3': 'N3期',
            'REM': 'REM期'
        }
        for stage, count in stage_counts.items():
            stage_cn = stage_names_cn.get(stage, stage)
            print(f"  {stage_cn}: {count} 个注释")

    # 呼吸事件信息
    resp_events = data['respiratory_events']
    print("\n呼吸事件信息:")
    print(f"呼吸事件总数: {len(resp_events)}")

    # 按类型统计
    apnea_events = [e for e in resp_events if e['type'] == 'apnea']
    hypopnea_events = [e for e in resp_events if e['type'] == 'hypopnea']
    print(f"呼吸暂停事件: {len(apnea_events)} 个")
    print(f"低通气事件: {len(hypopnea_events)} 个")

    # 持续时间统计
    if resp_events:
        apnea_durations = [e['duration_seconds'] for e in apnea_events]
        hypopnea_durations = [e['duration_seconds'] for e in hypopnea_events]

        if apnea_durations:
            print(f"呼吸暂停持续时间 (秒): 最小={min(apnea_durations):.2f}, 最大={max(apnea_durations):.2f}, 平均={sum(apnea_durations)/len(apnea_durations):.2f}")
        if hypopnea_durations:
            print(f"低通气持续时间 (秒): 最小={min(hypopnea_durations):.2f}, 最大={max(hypopnea_durations):.2f}, 平均={sum(hypopnea_durations)/len(hypopnea_durations):.2f}")

        # 显示事件的详细时间信息
        print("\n事件详细信息:")
        print("序号\t类型\t\t开始时间(小时)\t结束时间(小时)\t持续时间(秒)")
        print("-" * 70)

        for i, event in enumerate(sorted(resp_events, key=lambda e: e['start_time'])):
            event_type = "呼吸暂停" if event['type'] == 'apnea' else "低通气"
            start_hour = event['start_time'] / 3600
            end_hour = event['end_time'] / 3600
            duration = event['duration_seconds']
            print(f"{i+1}\t{event_type}\t{start_hour:.2f}\t\t{end_hour:.2f}\t\t{duration:.2f}")

    # AHI计算
    stats = data['statistics']
    print(f"\n呼吸暂停低通气指数 (AHI): {stats['ahi']:.2f}")

    # 保存详细分析结果到文本文件
    logger.info(f"保存分析结果到 {output_text_file}")
    with open(output_text_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"文件分析: {os.path.basename(args.file)}\n")
        f.write("=" * 80 + "\n\n")

        # 记录信息
        record_info = data['record_info']
        f.write("记录信息:\n")
        f.write(f"记录名称: {record_info['record_name']}\n")
        f.write(f"采样率: {record_info['fs']} Hz\n")
        f.write(f"样本数量: {record_info['n_samples']}\n")
        if 'n_signals' in record_info:
            f.write(f"信号数量: {record_info['n_signals']}\n")
        if 'signal_names' in record_info:
            f.write(f"信号名称: {', '.join(record_info['signal_names'])}\n")
        f.write(f"持续时间: {record_info['duration_seconds']:.2f} 秒 ({record_info['duration_hours']:.2f} 小时)\n\n")

        # 注释信息
        ann_info = data['annotation_info']
        f.write("注释信息:\n")
        f.write(f"总注释数: {ann_info['total_annotations']}\n")
        f.write(f"唯一符号: {', '.join(ann_info['unique_symbols'])}\n")
        if ann_info['unique_aux_notes']:
            f.write(f"唯一辅助注释: {', '.join(ann_info['unique_aux_notes'])}\n\n")

        # 睡眠阶段信息
        sleep_stages = data['sleep_stages']
        f.write("睡眠阶段信息:\n")
        f.write(f"睡眠阶段注释总数: {len(sleep_stages)}\n")
        if sleep_stages:
            # 统计每个阶段的出现次数
            stage_counts = {}
            for stage in sleep_stages:
                stage_name = stage['stage']
                if stage_name not in stage_counts:
                    stage_counts[stage_name] = 0
                stage_counts[stage_name] += 1

            f.write("睡眠阶段分布:\n")
            stage_names_cn = {
                'Wake': '清醒',
                'N1': 'N1期',
                'N2': 'N2期',
                'N3': 'N3期',
                'REM': 'REM期'
            }
            for stage, count in stage_counts.items():
                stage_cn = stage_names_cn.get(stage, stage)
                f.write(f"  {stage_cn}: {count} 个注释\n")
        f.write("\n")

        # 呼吸事件信息
        resp_events = data['respiratory_events']
        f.write("呼吸事件信息:\n")
        f.write(f"呼吸事件总数: {len(resp_events)}\n")

        # 按类型统计
        apnea_events = [e for e in resp_events if e['type'] == 'apnea']
        hypopnea_events = [e for e in resp_events if e['type'] == 'hypopnea']
        f.write(f"呼吸暂停事件: {len(apnea_events)} 个\n")
        f.write(f"低通气事件: {len(hypopnea_events)} 个\n\n")

        # 持续时间统计
        if resp_events:
            apnea_durations = [e['duration_seconds'] for e in apnea_events]
            hypopnea_durations = [e['duration_seconds'] for e in hypopnea_events]

            if apnea_durations:
                f.write(f"呼吸暂停持续时间 (秒): 最小={min(apnea_durations):.2f}, 最大={max(apnea_durations):.2f}, 平均={sum(apnea_durations)/len(apnea_durations):.2f}\n")
            if hypopnea_durations:
                f.write(f"低通气持续时间 (秒): 最小={min(hypopnea_durations):.2f}, 最大={max(hypopnea_durations):.2f}, 平均={sum(hypopnea_durations)/len(hypopnea_durations):.2f}\n\n")

            # 显示事件的详细时间信息
            f.write("事件详细信息:\n")
            f.write("序号\t类型\t\t开始时间(小时)\t结束时间(小时)\t持续时间(秒)\n")
            f.write("-" * 70 + "\n")

            for i, event in enumerate(sorted(resp_events, key=lambda e: e['start_time'])):
                event_type = "呼吸暂停" if event['type'] == 'apnea' else "低通气"
                start_hour = event['start_time'] / 3600
                end_hour = event['end_time'] / 3600
                duration = event['duration_seconds']
                f.write(f"{i+1}\t{event_type}\t{start_hour:.2f}\t\t{end_hour:.2f}\t\t{duration:.2f}\n")

        # AHI计算
        stats = data['statistics']
        f.write(f"\n呼吸暂停低通气指数 (AHI): {stats['ahi']:.2f}\n")

    print(f"\n详细分析结果已保存到: {output_text_file}")

    # Visualize events
    logger.info(f"Generating visualization to {output_file}")
    # Create a copy of data with English-only record name to avoid Chinese characters in plot
    plot_data = data.copy()
    plot_data['record_info'] = data['record_info'].copy()
    plot_data['record_info']['record_name'] = record_name  # Use simple record name without Chinese characters

    visualize_events(plot_data, output_file)
    print(f"可视化图表已保存到: {output_file}")

if __name__ == "__main__":
    main()
