import numpy as np# 数值计算库
import mne # 脑电图处理库
from mne.datasets.sleep_physionet.age import fetch_data# MNE中的数据获取工具

from tqdm import tqdm  # 进度条工具
from utils import *  # 导入自定义工具函数
import multiprocessing  # 多进程处理
from typing import List, Tuple  # 类型提示
import csv  # CSV文件处理
from loguru import logger  # 日志工具
import warnings  # 警告控制
import pickle  # Python对象序列化库

import sys
import config# 配置文件
from config import ALL_CHANNELS, PATH_TO_RAW_DATA, PATH_TO_PROCESSED_DATA# 从配置文件导入通道列表和路径
from scipy.io import loadmat# 加载MAT文件
import h5py# 处理HDF5文件
import glob# 文件路径处理
import scipy.signal# 信号处理
import os# 操作系统相关操作
import argparse

EVENT_TO_ID = {
    "wake": 1, 
    "nonrem1": 2, #非快速眼动期1
    "nonrem2": 3, #非快速眼动期2
    "nonrem3": 4, #非快速眼动期3
    "rem": 5, #快速眼动期
}
ID_TO_EVENT = {value: key for key, value in EVENT_TO_ID.items()}# 将事件ID转换为事件名称


# 忽略所有警告
warnings.filterwarnings("ignore")

def import_signal_names(file_name):
    """
    从.hea头文件中提取信号名称、采样率和样本数信息
    参数:
        file_name: 头文件路径    
    返回:
        s: 信号名称列表
        Fs: 采样率
        n_samples: 样本数
    """
    with open(file_name, 'r') as myfile:
        s = myfile.read()
        s = s.split('\n')
        s = [x.split() for x in s]

        n_signals = int(s[0][1])# 信号数量
        n_samples = int(s[0][3])# 样本数
        Fs        = int(s[0][2])# 采样率

        s = s[1:-1]# 去除头尾
        s = [s[i][8] for i in range(0, n_signals)]
    return s, Fs, n_samples

def extract_labels(path):
    """
    从特定格式的.mat文件中提取标签信息
    
    参数:
        path: .mat文件路径
        
    返回:
        labels: 标签数组
        label_names: 标签名称列表
    """
    data = h5py.File(path, 'r')# 读取HDF5格式的mat文件
    length = data['data']['sleep_stages']['wake'].shape[1]# 获取标签长度
    labels = np.zeros((length, 6)) # 初始化标签数组
    #填充标签   
    for i, label in enumerate(data['data']['sleep_stages'].keys()):
        labels[:,i] = data['data']['sleep_stages'][label][:]
    
    return labels, list(data['data']['sleep_stages'].keys())  # 返回标签数组和标签名称列表

def resample_signal(data, labels, old_fs):
    """
    重采样信号和对应的标签
    
    参数:
        data: 原始数据
        labels: 原始标签
        old_fs: 原始采样率
        
    返回:
        resampled_data: 重采样后的数据
        resampled_labels: 重采样后的标签
    """
    # 找到标签变化点，进行剪切
    diff = np.diff(labels, axis = 0)
    cutoff = np.where(diff[:,4] != 0)[0]+1
    data, labels = data[cutoff[0]+1:,:], labels[cutoff[0]+1:,:]
    #重新采样到100Hz
    new_fs = 100
    num = int(len(data)/(old_fs/new_fs))
    resampled_data = scipy.signal.resample(data, num = num, axis = 0)
    resampled_labels = labels[::int((old_fs/new_fs)),:]
    return resampled_data.astype(np.int16), resampled_labels.astype(np.int16)


def preprocess_EEG(folder,remove_files = False, out_folder = None):
    """
    预处理EEG数据，将原始数据转换为MNE格式
    
    参数:
        folder: 数据文件夹
        remove_files: 是否删除原始文件
        out_folder: 输出文件夹
    """
    files = glob.glob(f'{folder}/*')# 获取文件夹中的所有文件    
    data = None
    labels = None
    Fs = None
    # 遍历文件夹中的所有文件
    for file in files:
        if '.hea' in file:# 如果是头文件
            s, Fs, n_samples = import_signal_names(file)# 提取信号名称、采样率和样本数
            if remove_files:# 如果需要删除原始文件
                os.remove(file)# 删除头文件
        elif '-arousal.mat' in file:# 如果是 arousal.mat 文件
            labels, label_names = extract_labels(file)# 提取标签
            if remove_files:# 如果需要删除原始文件
                os.remove(file)
        elif 'mat' in file:# 如果是 mat 文件
            data = loadmat(file)['val'][:6, :]# 提取数据
            if remove_files:# 如果需要删除原始文件
                os.remove(file)
    # 如果数据不为空,进行处理
    if not data is None:
        # 找到标签变化点，进行剪切
        diff = np.diff(labels, axis = 0)
        cutoff = np.where(diff[:,4] != 0)[0]+1# 找到标签变化点
        data, labels = data[:, cutoff[0]+1:], labels[cutoff[0]+1:,:]# 剪切数据和标签
        # 创建MNE信息
        info = mne.create_info(s[:6], Fs, ch_types = 'eeg')
        mne_dataset = mne.io.RawArray(data, info)# 创建MNE数据集
        # 处理标签
        events = process_labels_to_events(labels, label_names)# 处理标签
        label_dict = dict(zip(np.arange(0,6), label_names))# 创建标签字典
        events = np.array(events)# 转换为数组
        event_dict = dict(zip(label_names, np.arange(0,6)))# 创建事件字典
        f = lambda x: label_dict[x]# 定义标签转换函数
        annotations = mne.Annotations(onset = events[:,0]/Fs, duration = events[:,1]/Fs, description  = list(map(f,events[:,2])))# 创建注释
        mne_dataset.set_annotations(annotations)# 设置注释
        # 重采样到100Hz
        mne_dataset.resample(sfreq = 100)
        # 创建30s片段的事件
        epoch_events = mne.events_from_annotations(mne_dataset, chunk_duration = 30)
        # 添加刺激通道
        info = mne.create_info(['STI'], mne_dataset.info['sfreq'], ['stim'])
        stim_data = np.zeros((1, len(mne_dataset.times)))
        stim_raw = mne.io.RawArray(stim_data, info)# 创建刺激数据   
        mne_dataset.add_channels([stim_raw], force_update_info=True)
        mne_dataset.add_events(epoch_events[0], stim_channel = 'STI')# 添加刺激事件         
        mne_dataset.save(f'{out_folder}/001_30s_raw.fif', overwrite = True)# 保存处理后的数据

def relocate_EEG_data(folder, remove_files = True):
    """
    重定位EEG数据，将处理后的数据移动到新位置
    
    参数:
        folder: 文件夹路径
        remove_files: 是否删除原始文件
    """
    data_file = mne.read_epochs(f'{folder}/001_30s.fif')# 读取处理后的数据
    #h5py.File(f'{folder}/data.hdf5', 'r')
    new_name = f'{folder}/001_30s_epo.fif'# 新文件名
    data_file.save(new_name)# 保存处理后的数据
    if remove_files:# 如果需要删除原始文件
        os.remove(f'{folder}/data.mat')# 删除原始文件
        os.remove(f'{folder}/001_30s.fif')# 删除原始文件

def process_labels_to_events(labels, label_names):
    """
    将标签转换为事件
    
    参数:
        labels: 标签数组
        label_names: 标签名称列表
    返回:
        events: 事件列表 [开始时间, 持续时间, 标签]
    """
    new_labels = np.argmax(labels, axis = 1)# 获取每个时间点的最大标签值
    lab = new_labels[0]# 初始标签
    events = []# 初始事件列表
    start = 0# 初始开始时间
    i = 0# 初始索引
    while i < len(new_labels)-1:# 遍历标签
        while new_labels[i] == lab and i < len(new_labels)-1:# 如果标签相同
            i+=1
        end = i
        dur = end +1 - start# 计算持续时间
        events.append([start, dur, lab])# 添加事件
        lab = new_labels[i]# 更新标签
        start = i+1# 更新开始时间
    return events


def get_arguments():
    """
    解析命令行参数
    
    返回:
        args: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="Process data and save to files")# 创建解析器
    parser.add_argument("--data_path", type=str, default=None, 
                        help="Path to the EDF files")# 添加数据路径参数         
    parser.add_argument("--save_path", type=str, default=None, 
                        help="Path to save preprocessed data")# 添加保存路径参数
    parser.add_argument("--num_files", type=int, default=10, 
                        help="Number of files to process")#处理文件数量
    parser.add_argument("--chunk_duration", type=float, default=30.0,
                        help="Duration of data chunks in seconds")#数据块时长（秒）
    parser.add_argument("--num_threads", type=int, default=4,
                            help="Number of threads for parallel processing")#线程数量
    parser.add_argument("--target_sampling_rate", type=int, default=256,
                            help="Target Sampling of the dataset")#目标采样率
          
    return parser.parse_args()

def parallel_process_edf_file(args: Tuple[List[str], str, float]):
    """
    并行处理EDF文件
    
    参数:
        args: 元组，包含(EDF文件列表, 保存路径, 块持续时间, 目标采样率)
    
    返回:
        [1]: 表示处理完成的标记
    """
    edf_and_event_files = args[0]# EDF文件列表
    path_to_save = args[1]# 保存路径
    chunk_duration = args[2]# 块持续时间
    target_sampling_rate = args[3]# 目标采样率

    path_to_X = os.path.join(path_to_save, "X")# 创建X文件夹
    path_to_Y = os.path.join(path_to_save, "Y")# 创建Y文件夹

    for edf_dict in tqdm(edf_and_event_files):# 遍历EDF文件列表

        arousal_mat_filename = edf_dict["arousal_mat"]# arousal.mat文件路径
        hea_filename = edf_dict["hea"]# hea文件路径
        edf_filename = edf_dict["mat"]# mat文件路径

        # 使用os.path.basename获取文件名，并去掉扩展名以获取前缀
        file_prefix = os.path.basename(edf_filename).split('.')[0]

        path_to_patient_X = os.path.join(path_to_X, file_prefix)# 创建X文件夹路径
        path_to_patient_Y = os.path.join(path_to_Y, f"{file_prefix}.pickle")# 创建Y文件夹路径
        #如果X文件夹存在且不为空，则跳过处理
        if os.path.exists(path_to_patient_X) and len(os.listdir(path_to_patient_X)) > 0:
            logger.info(f"Patient already processed: {path_to_patient_X}")
            continue

        try:#加载数据和标签
            s, Fs, n_samples = import_signal_names(hea_filename)# 导入信号名称、采样率和样本数
            labels, label_names = extract_labels(arousal_mat_filename)# 提取标签
            data = loadmat(edf_filename)['val']# 加载数据
        except:
            continue

        if data is None:
            continue

        try:# 处理数据和标签    
            diff = np.diff(labels, axis = 0)# 计算标签差值
            cutoff = np.where(diff[:,4] != 0)[0]+1# 找到标签变化点
            data, labels = data[:, cutoff[0]+1:], labels[cutoff[0]+1:,:]# 剪切数据和标签
            info = mne.create_info(s, Fs, ch_types = 'eeg')# 创建MNE信息
            edf_raw = mne.io.RawArray(data, info)# 创建MNE数据集

            events = process_labels_to_events(labels, label_names)
            label_dict = dict(zip(np.arange(0,6), label_names))
            events = np.array(events)
            event_dict = dict(zip(label_names, np.arange(0,6)))
            f = lambda x: label_dict[x]
            annotations = mne.Annotations(onset = events[:,0]/Fs, duration = events[:,1]/Fs, description  = list(map(f,events[:,2])))
            edf_raw.set_annotations(annotations, emit_warning=True)
            
            edf_raw.resample(sfreq = 256)# 重采样到256Hz

            sfreq = edf_raw.info["sfreq"]
            logger.info(f"Original sfreq: {sfreq}")
            # 从注释中提取事件
            events_raw, _ = mne.events_from_annotations(edf_raw, event_id=EVENT_TO_ID, chunk_duration=chunk_duration)
            event_ids = set(events_raw[:, 2])
            true_event_to_id = {ID_TO_EVENT[idx]: idx for idx in event_ids}
            tmax = chunk_duration - 1.0 / sfreq  # tmax in included
            # 创建epochs    
            epochs = mne.Epochs(
                raw=edf_raw,# 原始数据  
                events=events_raw,# 事件
                event_id=true_event_to_id,# 事件ID
                tmin=0.0,# 事件开始时间
                tmax=tmax,# 事件结束时间
                baseline=None,# 基线
                event_repeated='drop',# 重复事件处理
            ).drop_bad()# 丢弃坏的epochs
        except Exception as e:
            print(f"Warning: An error occurred - {e}")
            continue
        # 找出edf_raw中与ALL_CHANNELS匹配的通道索引 
        indices = [edf_raw.ch_names.index(channel) for channel in ALL_CHANNELS if channel in edf_raw.ch_names]
    
        labels = {}
        # 创建保存目录
        path_to_patient_X = os.path.join(path_to_X, file_prefix)
        path_to_patient_Y = os.path.join(path_to_Y, f"{file_prefix}.pickle")

        os.makedirs(path_to_patient_X, exist_ok=True)
        #处理每一个epoch    
        for idx, epoch in enumerate(epochs):
            data = epoch[indices, :]
            assert data.shape[0] == len(indices)
            #检查数据有效性
            if data.shape[0] != len(ALL_CHANNELS) or data.shape[1] == 0:
                continue
            #重采样到目标采样率
            num_samples_target = int(data.shape[1] * target_sampling_rate / sfreq)
            resampled_data = scipy.signal.resample(data, num_samples_target, axis=1)
            #保存为npy文件
            file_name = f"{file_prefix}_{idx}.npy"
            np.save(os.path.join(path_to_patient_X, file_name), resampled_data)
            #保存标签
            labels[file_name] = epochs[idx].event_id

        with open(path_to_patient_Y, 'wb') as file:
            pickle.dump(labels, file)

    return [1]#返回成功标志


def main():
    """
    主函数，处理命令行参数并启动数据处理
    """
    args = get_arguments()  # 获取命令行参数

    path_to_edf_files = args.data_path  # 数据路径
    path_to_save = args.save_path  # 保存路径

    # 如果未指定路径，使用配置文件中的默认路径
    if path_to_edf_files == None:
        path_to_edf_files = PATH_TO_RAW_DATA
    if path_to_save == None:
        path_to_save = PATH_TO_PROCESSED_DATA
        
    # 检查路径是否存在
    if not os.path.exists(path_to_edf_files):
        logger.error(f"数据路径不存在: {path_to_edf_files}")
        return
        
    logger.info(f"使用数据路径: {path_to_edf_files}")
    logger.info(f"使用保存路径: {path_to_save}")

    # 获取其他参数
    num_files = args.num_files  # 处理文件数
    chunk_duration = args.chunk_duration  # 块持续时间
    num_threads = args.num_threads  # 线程数
    target_sampling_rate = args.target_sampling_rate  # 目标采样率
    
    logger.info(f"处理文件数: {num_files if num_files != -1 else '全部'}")
    logger.info(f"使用线程数: {num_threads}")

    # 创建保存目录
    os.makedirs(path_to_save, exist_ok=True)

    path_to_X = os.path.join(path_to_save, "X")  # 数据保存路径
    path_to_Y = os.path.join(path_to_save, "Y")  # 标签保存路径

    os.makedirs(path_to_X, exist_ok=True)
    os.makedirs(path_to_Y, exist_ok=True)

    data_dict = {}  # 初始化数据字典

    # 获取所有受试者
    subjects = os.listdir(path_to_edf_files)
    subjects_path = [os.path.join(path_to_edf_files, subject) for subject in subjects]

    # 筛选有效的EDF文件
    edf_events_files_pruned = []
    for subject_path in subjects_path:
        # 跳过非目录文件和特殊目录
        if not os.path.isdir(subject_path) or "RECORDS" in subject_path or "ANNOTATORS" in subject_path:
            continue
            
        # 文件数量检查，我们不再严格要求4个文件
        # 而是检查必要的文件类型是否存在
        files = glob.glob(f'{subject_path}/*')
        try:
            temp_dict = {}
            has_required_files = False
            
            # 计数所需文件类型
            mat_file = False
            hea_file = False
            arousal_mat_file = False
            
            for file in files:
                # 识别不同类型的文件
                if '.hea' in file:
                    temp_dict["hea"] = os.path.join(file)
                    hea_file = True
                elif '-arousal.mat' in file:
                    temp_dict["arousal_mat"] = os.path.join(file)
                    arousal_mat_file = True
                elif '.mat' in file and '-arousal.mat' not in file:
                    temp_dict["mat"] = os.path.join(file)
                    mat_file = True
            
            # 只有当所有必要文件都存在时才添加
            if mat_file and hea_file and arousal_mat_file:
                edf_events_files_pruned.append(temp_dict)
                logger.info(f"找到有效数据目录: {subject_path}")
            else:
                logger.warning(f"目录缺少必要文件: {subject_path}")
        except Exception as e:
            logger.error(f"处理目录时出错: {subject_path}, 错误: {str(e)}")
            continue

    logger.info(f"Starting Extraction...")  # 开始提取数据

    # 如果指定了文件数量限制，则取前num_files个
    if num_files != -1:
        edf_events_files_pruned = edf_events_files_pruned[:num_files]

    # 将文件划分给不同线程
    edf_and_event_files_per_thread = np.array_split(edf_events_files_pruned, num_threads)

    # 创建任务列表
    tasks = [(edf_and_event_file, path_to_save, chunk_duration, target_sampling_rate) for edf_and_event_file in edf_and_event_files_per_thread]
    
    # 使用多进程并行处理
    with multiprocessing.Pool(num_threads) as pool:
        preprocessed_results = [
            y for x in pool.imap_unordered(parallel_process_edf_file, tasks) for y in x
        ]

if __name__ == "__main__":
    main()  # 执行主函数