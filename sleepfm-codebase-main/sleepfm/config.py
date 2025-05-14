"""设置基于配置文件的路径。"""

import configparser
import os
import types
#配置文件名和参数字典的初始化
_FILENAME = None
_PARAM = {}

#配置文件的命名空间
CONFIG = types.SimpleNamespace(
    FILENAME=_FILENAME,
    DATASETS=_PARAM.get("datasets", "datasets"),#数据集
    OUTPUT=_PARAM.get("output", "output"),#输出
    CACHE=_PARAM.get("cache", ".cache"),#缓存
)

# 定义原始数据和处理后数据的路径
PATH_TO_RAW_DATA = "/root/autodl-tmp/physionet.org/files/challenge-2018/1.0.0/training/"# 原始CinC数据集路径
PATH_TO_PROCESSED_DATA = "/root/autodl-tmp/processed_data/pc18"# 处理后数据存放路径

# 定义SleepFM框架的通用数据和模型路径
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
PHYSIONET_PATH = PATH_TO_RAW_DATA

# Define Sleep related global variables睡眠相关的全局变量
# 睡眠阶段标签到数字的映射字典（用于分类任务）
LABELS_DICT = {
    "Wake": 0,
    "Stage 1": 1,
    "Stage 2": 2,
    "Stage 3": 3,
    "REM": 4
}
# 模型使用的模态类型列表
MODALITY_TYPES = ["respiratory", "sleep_stages", "ekg"]  # 呼吸信号、睡眠脑电、心电图信号
CLASS_LABELS = ["Wake", "Stage 1", "Stage 2", "Stage 3", "REM"]  # 分类标签名称
NUM_CLASSES = 5  # 分类类别数量

# 睡眠事件到ID的映射（用于事件标注）
EVENT_TO_ID = {
    "Wake": 1,     # 清醒
     "Stage 1": 2, # 1期睡眠
     "Stage 2": 3, # 2期睡眠
     "Stage 3": 4, # 3期睡眠
     "Stage 4": 4, # 4期睡眠（与3期合并）
     "REM": 5,     # 快速眼动睡眠
}

# 不同标注格式的睡眠阶段标签统一映射
LABEL_MAP = {
    # AASM格式标签
    "Sleep stage W": "Wake",    # 清醒
    "Sleep stage N1": "Stage 1", # 非快速眼动1期
    "Sleep stage N2": "Stage 2", # 非快速眼动2期
    "Sleep stage N3": "Stage 3", # 非快速眼动3期（慢波睡眠）
    "Sleep stage R": "REM",      # 快速眼动睡眠
    # 简写格式
    "W": "Wake",
    "N1": "Stage 1",
    "N2": "Stage 2",
    "N3": "Stage 3",
    "REM": "REM",
    # 其他可能的格式
    "wake": "Wake",
    "nonrem1": "Stage 1",
    "nonrem2": "Stage 2",
    "nonrem3": "Stage 3",
    "rem": "REM",
}

# 定义数据集中的所有通道
ALL_CHANNELS = ['F3-M2',    # 额叶脑电-左
 'F4-M1',                   # 额叶脑电-右
 'C3-M2',                   # 中央脑电-左
 'C4-M1',                   # 中央脑电-右
 'O1-M2',                   # 枕叶脑电-左
 'O2-M1',                   # 枕叶脑电-右
 'E1-M2',                   # 眼电图
 'Chin1-Chin2',             # 下颌肌电图
 'ABD',                     # 腹部呼吸带
 'CHEST',                   # 胸部呼吸带
 'AIRFLOW',                 # 气流
 'SaO2',                    # 血氧饱和度
 'ECG']                     # 心电图



# 按不同模态分组的通道数据
CHANNEL_DATA = {
    "Respiratory": ["CHEST", "SaO2", "ABD"],  # 呼吸相关通道
    "Sleep_Stages": ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2"],  # 睡眠脑电相关通道
    "EKG": ["ECG"],  # 心电图通道
    }

# 各模态对应通道在ALL_CHANNELS中的索引
CHANNEL_DATA_IDS = {
    "Respiratory": [ALL_CHANNELS.index(item) for item in CHANNEL_DATA["Respiratory"]], #呼吸相关通道在ALL_CHANNELS中的索引
    "Sleep_Stages": [ALL_CHANNELS.index(item) for item in CHANNEL_DATA["Sleep_Stages"]], #睡眠脑电相关通道在ALL_CHANNELS中的索引
    "EKG": [ALL_CHANNELS.index(item) for item in CHANNEL_DATA["EKG"]], #心电图通道在ALL_CHANNELS中的索引
 }

# SDB（睡眠呼吸障碍）相关配置
SDB_CONFIG = {
    "window_size": 30,  # 窗口大小（秒）
    "event_threshold": 0.3,  # 窗口内SDB事件占比超过此阈值则标记为阳性样本
    "respiratory_channels": ["CHEST", "ABD", "SaO2"],  # SDB检测使用的呼吸相关通道
    "sample_rate": 256,  # 采样率（与改进版预处理脚本一致）
    "class_names": ["Non-SDB", "SDB"],  # 分类类别
    "event_types": {
        "apnea": ["apnea", "apnoea", "obstructive apnea", "central apnea", "mixed apnea"],  # 呼吸暂停事件关键词
        "hypopnea": ["hypopnea", "hypopoea", "obstructive hypopnea"]  # 低通气事件关键词
    },
    "event_ids": {
        "non_sdb": 0,  # 非SDB事件ID
        "apnea": 1,    # 呼吸暂停事件ID
        "hypopnea": 2  # 低通气事件ID
    },
    "data_path": "/root/autodl-tmp/processed_data/pc18/sdb",  # SDB数据保存路径
    "raw_data_path": "/root/autodl-tmp/physionet.org/files/challenge-2018/1.0.0/training"  # 原始数据路径
}

# SDB严重程度分类标准（基于AHI）
SDB_SEVERITY = {
    "normal": (0, 5),       # 正常: AHI < 5
    "mild": (5, 15),        # 轻度: 5 ≤ AHI < 15
    "moderate": (15, 30),   # 中度: 15 ≤ AHI < 30
    "severe": (30, float('inf'))  # 重度: AHI ≥ 30
}

# Config类定义
class Config:
    """配置类，提供对项目配置的访问"""

    def __init__(self):
        # 数据路径
        self.DATA_PATH = DATA_PATH
        self.MODEL_PATH = MODEL_PATH
        self.PHYSIONET_PATH = PHYSIONET_PATH

        # SDB相关配置
        self.SDB_CONFIG = SDB_CONFIG
        self.SDB_SEVERITY = SDB_SEVERITY