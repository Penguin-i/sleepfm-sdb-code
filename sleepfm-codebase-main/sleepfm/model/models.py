from torch import nn
import torch
import torch.nn.functional as F
from loguru import logger

from torch import nn
from collections import OrderedDict


class Bottleneck(nn.Module):
    def __init__(self,in_channel,out_channel,expansion,activation,stride=1,padding = 1):
        super(Bottleneck, self).__init__()
        self.stride=stride
        self.conv1 = nn.Conv1d(in_channel,in_channel*expansion,kernel_size = 1)
        self.conv2 = nn.Conv1d(in_channel*expansion,in_channel*expansion,kernel_size = 3, groups = in_channel*expansion,
                               padding=padding,stride = stride)
        self.conv3 = nn.Conv1d(in_channel*expansion,out_channel,kernel_size = 1, stride =1)
        self.b0 = nn.BatchNorm1d(in_channel*expansion)
        self.b1 =  nn.BatchNorm1d(in_channel*expansion)
        self.d = nn.Dropout()
        self.act = activation()
    def forward(self,x):
        if self.stride == 1:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            y = self.d(y)
            y = x+y
            return y
        else:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            return y

from torch import nn
from collections import OrderedDict

class MBConv(nn.Module):
    def __init__(self,in_channel,out_channels,expansion,layers,activation=nn.ReLU6,stride = 2):
        super(MBConv, self).__init__()
        self.stack = OrderedDict()
        for i in range(0,layers-1):
            self.stack['s'+str(i)] = Bottleneck(in_channel,in_channel,expansion,activation)
            #self.stack['a'+str(i)] = activation()
        self.stack['s'+str(layers+1)] = Bottleneck(in_channel,out_channels,expansion,activation,stride=stride)
        # self.stack['a'+str(layers+1)] = activation()
        self.stack = nn.Sequential(self.stack)

        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self,x):
        x = self.stack(x)
        return self.bn(x)


"""def MBConv(in_channel,out_channels,expansion,layers,activation=nn.ReLU6,stride = 2):
    stack = OrderedDict()
    for i in range(0,layers-1):
        stack['b'+str(i)] = Bottleneck(in_channel,in_channel,expansion,activation)
    stack['b'+str(layers)] = Bottleneck(in_channel,out_channels,expansion,activation,stride=stride)
    return nn.Sequential(stack)"""


class EffNet(nn.Module):

    def __init__(
            self,
            in_channel,
            num_additional_features = 0,
            depth = [1,2,2,3,3,3,3],
            channels = [32,16,24,40,80,112,192,320,1280],
            dilation = 1,
            stride = 2,
            expansion = 6):
        super(EffNet, self).__init__()
        logger.info(f"depth: {depth}")
        self.stage1 = nn.Conv1d(in_channel, channels[0], kernel_size=3, stride=stride, padding=1,dilation = dilation) #1 conv
        self.b0 = nn.BatchNorm1d(channels[0])
        self.stage2 = MBConv(channels[0], channels[1], expansion, depth[0], stride=2)# 16 #input, output, depth # 3 conv
        self.stage3 = MBConv(channels[1], channels[2], expansion, depth[1], stride=2)# 24 # 4 conv # d 2
        self.Pool = nn.MaxPool1d(3, stride=1, padding=1) #
        self.stage4 = MBConv(channels[2], channels[3], expansion, depth[2], stride=2)# 40 # 4 conv # d 2
        self.stage5 = MBConv(channels[3], channels[4], expansion, depth[3], stride=2)# 80 # 5 conv # d
        self.stage6 = MBConv(channels[4], channels[5], expansion, depth[4], stride=2)# 112 # 5 conv
        self.stage7 = MBConv(channels[5], channels[6], expansion, depth[5], stride=2)# 192 # 5 conv
        self.stage8 = MBConv(channels[6], channels[7], expansion, depth[6], stride=2)# 320 # 5 conv

        self.stage9 = nn.Conv1d(channels[7], channels[8], kernel_size=1)
        self.AAP = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
        self.num_additional_features = num_additional_features
        self.fc = nn.Linear(channels[8] + num_additional_features, 1)


    def forward(self, x):
        if self.num_additional_features >0:
            x,additional = x
        # N x 12 x 2500
        x = self.b0(self.stage1(x))
        # N x 32 x 1250
        x = self.stage2(x)
        # N x 16 x 625
        x = self.stage3(x)
        # N x 24 x 313
        x = self.Pool(x)
        # N x 24 x 313

        x = self.stage4(x)
        # N x 40 x 157
        x = self.stage5(x)
        # N x 80 x 79
        x = self.stage6(x)
        # N x 112 x 40
        x = self.Pool(x)
        # N x 192 x 20

        x = self.stage7(x)
        # N x 320 x 10
        x = self.stage8(x)
        x = self.stage9(x)
        # N x 1280 x 10
        x = self.act(self.AAP(x)[:,:,0])
        # N x 1280
        x = self.drop(x)
        if self.num_additional_features >0:
            x = torch.cat((x,additional),1)
        x = self.fc(x)
        # N x 1
        return x


class EffNetSupervised(nn.Module):

    def __init__(
            self,
            in_channel,
            num_classes = 5,
            num_additional_features = 0,
            depth = [1,2,2,3,3,3,3],
            channels = [32,16,24,40,80,112,192,320,1280],
            dilation = 1,
            stride = 2,
            expansion = 6):
        super(EffNetSupervised, self).__init__()
        logger.info(f"depth: {depth}")
        self.stage1 = nn.Conv1d(in_channel, channels[0], kernel_size=3, stride=stride, padding=1,dilation = dilation) #1 conv
        self.b0 = nn.BatchNorm1d(channels[0])
        self.stage2 = MBConv(channels[0], channels[1], expansion, depth[0], stride=2)# 16 #input, output, depth # 3 conv
        self.stage3 = MBConv(channels[1], channels[2], expansion, depth[1], stride=2)# 24 # 4 conv # d 2
        self.Pool = nn.MaxPool1d(3, stride=1, padding=1) #
        self.stage4 = MBConv(channels[2], channels[3], expansion, depth[2], stride=2)# 40 # 4 conv # d 2
        self.stage5 = MBConv(channels[3], channels[4], expansion, depth[3], stride=2)# 80 # 5 conv # d
        self.stage6 = MBConv(channels[4], channels[5], expansion, depth[4], stride=2)# 112 # 5 conv
        self.stage7 = MBConv(channels[5], channels[6], expansion, depth[5], stride=2)# 192 # 5 conv
        self.stage8 = MBConv(channels[6], channels[7], expansion, depth[6], stride=2)# 320 # 5 conv

        self.stage9 = nn.Conv1d(channels[7], channels[8], kernel_size=1)
        self.AAP = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
        self.num_additional_features = num_additional_features
        self.fc = nn.Linear(channels[8] + num_additional_features, num_classes)

    def forward(self, x):
        if self.num_additional_features >0:
            x,additional = x
        # N x 12 x 2500
        x = self.b0(self.stage1(x))
        # N x 32 x 1250
        x = self.stage2(x)
        # N x 16 x 625
        x = self.stage3(x)
        # N x 24 x 313
        x = self.Pool(x)
        # N x 24 x 313

        x = self.stage4(x)
        # N x 40 x 157
        x = self.stage5(x)
        # N x 80 x 79
        x = self.stage6(x)
        # N x 112 x 40
        x = self.Pool(x)
        # N x 192 x 20

        x = self.stage7(x)
        # N x 320 x 10
        x = self.stage8(x)
        x = self.stage9(x)
        # N x 1280 x 10
        x = self.act(self.AAP(x)[:,:,0])
        # N x 1280
        x = self.drop(x)
        if self.num_additional_features >0:
            x = torch.cat((x,additional),1)
        x = self.fc(x)
        # x = F.log_softmax(x, dim=1)
        return x


class MultiModalSDBClassifier(nn.Module):
    """多模态SDB检测分类器，支持输入多种生理信号"""

    def __init__(self, modality_models, embedding_dim=512, fusion_method='concat', class_weights=None):
        """
        初始化

        参数:
        modality_models: 各模态对应的特征提取模型字典，如
                        {'Respiratory': resp_model, 'EKG': ekg_model, 'Sleep_Stages': sleep_model}
        embedding_dim: 各模态输出的嵌入维度
        fusion_method: 特征融合方法，可选['concat', 'attention', 'weighted']
        """
        super(MultiModalSDBClassifier, self).__init__()
        self.modality_models = nn.ModuleDict(modality_models)
        self.modalities = list(modality_models.keys())
        self.embedding_dim = embedding_dim
        self.fusion_method = fusion_method
        self.class_weights = class_weights  # 用于处理类别不平衡

        # 计算融合后的特征维度
        if fusion_method == 'concat':
            fused_dim = embedding_dim * len(self.modalities)
        else:
            fused_dim = embedding_dim

        # 多层感知机(MLP)分类头
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
            # 移除Sigmoid激活函数，使用BCEWithLogitsLoss可以提高数值稳定性
        )

        # 如果使用注意力融合，添加注意力层
        if fusion_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

    def forward(self, x_dict):
        """
        前向传播

        参数:
        x_dict: 各模态的输入数据字典，如
               {'Respiratory': x_resp, 'EKG': x_ekg, 'Sleep_Stages': x_sleep}

        返回:
        输出概率和特征嵌入
        """
        # 提取每个模态的特征
        embeddings = {}
        for modality in self.modalities:
            if modality in x_dict:
                # 获取该模态的特征
                embeddings[modality] = self.modality_models[modality](x_dict[modality])
                # 归一化特征
                embeddings[modality] = F.normalize(embeddings[modality], dim=1)

        # 特征融合
        if self.fusion_method == 'concat':
            # 拼接所有模态的特征
            fused_embedding = torch.cat([embeddings[m] for m in self.modalities if m in embeddings], dim=1)

        elif self.fusion_method == 'attention':
            # 使用注意力机制融合
            attention_scores = {}
            for modality in embeddings:
                attention_scores[modality] = self.attention(embeddings[modality])

            # Softmax归一化注意力权重
            all_scores = torch.cat([attention_scores[m] for m in embeddings], dim=1)
            attention_weights = F.softmax(all_scores, dim=1)

            # 加权求和
            fused_embedding = torch.zeros(embeddings[self.modalities[0]].shape,
                                         device=embeddings[self.modalities[0]].device)

            for i, modality in enumerate(embeddings):
                fused_embedding += embeddings[modality] * attention_weights[:, i].unsqueeze(1)

        elif self.fusion_method == 'weighted':
            # 使用预定义权重融合
            weights = {
                'Respiratory': 1.3,  # 增加呼吸信号的权重
                'Sleep_Stages': 1.0,
                'EKG': 1.0
            }

            fused_embedding = torch.zeros(embeddings[self.modalities[0]].shape,
                                         device=embeddings[self.modalities[0]].device)

            weight_sum = 0
            for modality in embeddings:
                if modality in weights:
                    fused_embedding += embeddings[modality] * weights[modality]
                    weight_sum += weights[modality]

            # 归一化
            if weight_sum > 0:
                fused_embedding /= weight_sum

        else:
            # 默认使用第一个模态
            fused_embedding = embeddings[self.modalities[0]]

        # 分类
        logits = self.classifier(fused_embedding)

        # 返回logits而不是sigmoid输出，便于使用BCEWithLogitsLoss
        return logits, fused_embedding


class SDBFeatureExtractor(nn.Module):
    """用于提取SDB特征的模型"""

    def __init__(self, base_model, freeze_base=True):
        """
        初始化

        参数:
        base_model: 预训练的基础模型
        freeze_base: 是否冻结基础模型参数
        """
        super(SDBFeatureExtractor, self).__init__()
        self.base_model = base_model

        # 冻结基础模型参数
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        """前向传播"""
        with torch.no_grad():
            # 提取特征嵌入
            features = self.base_model(x)

            # 如果特征是模型的输出和嵌入的元组，只取嵌入
            if isinstance(features, tuple):
                features = features[1]  # 假设嵌入是第二个返回值

        return features


class SDBClassifierWithFocalLoss(nn.Module):
    """
    使用Focal Loss的SDB分类器，用于处理类别不平衡问题
    """

    def __init__(self, base_classifier, gamma=2.0, alpha=0.25):
        """
        初始化

        参数:
        base_classifier: 基础分类器模型
        gamma: Focal Loss的gamma参数，用于降低易分样本的权重
        alpha: Focal Loss的alpha参数，用于平衡正负样本
        """
        super(SDBClassifierWithFocalLoss, self).__init__()
        self.base_classifier = base_classifier
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, x_dict):
        """前向传播"""
        # 获取基础分类器的输出
        logits, embeddings = self.base_classifier(x_dict)
        return logits, embeddings

    def focal_loss(self, logits, targets):
        """
        计算Focal Loss

        参数:
        logits: 模型输出的logits
        targets: 目标标签

        返回:
        focal_loss: Focal Loss值
        """
        # 将logits转换为概率
        probs = torch.sigmoid(logits)

        # 计算二元交叉熵
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # 计算Focal Loss权重
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # 应用Focal Loss公式
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss

        return focal_loss.mean()


class SDBClassifierWithFreezeLayers(nn.Module):
    """
    带有冻结层的SDB分类器，用于微调
    """

    def __init__(self, base_classifier, freeze_layers=None):
        """
        初始化

        参数:
        base_classifier: 基础分类器模型
        freeze_layers: 要冻结的层名列表，如果为None则不冻结任何层
        """
        super(SDBClassifierWithFreezeLayers, self).__init__()
        self.base_classifier = base_classifier

        # 冻结指定层
        if freeze_layers is not None:
            for name, param in self.base_classifier.named_parameters():
                if any(layer_name in name for layer_name in freeze_layers):
                    param.requires_grad = False

    def forward(self, x_dict):
        """前向传播"""
        return self.base_classifier(x_dict)
